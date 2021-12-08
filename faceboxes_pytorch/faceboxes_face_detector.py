import os
import torch
import numpy as np
from .data import cfg
from .layers.functions.prior_box import PriorBox
from .models.faceboxes import FaceBoxes
from .utils.box_utils import decode, batch_decode, get_faceboxes_max_batch_size
from typing import List, Tuple

class FaceBoxesFaceDetector(object):
    def __init__(self, use_gpu=False):
        torch.set_grad_enabled(False)
        # net and model
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
        weight_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights/FaceBoxes.pth'))
        self.net = self.load_model(self.net, weight_path, True)
        self.net.eval()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.net = self.net.to(self.device)

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # print('Missing keys:{}'.format(len(missing_keys)))
        # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        # print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        # print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path, load_to_cpu):
        # print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            self.device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def get_faceboxes(self, image, threshold=0.2):
        resize = 1

        img = np.float32(image)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf = self.net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        if not boxes.any():
            return [], []

        scores_and_boxes = zip(scores, boxes)
        scores_and_boxes = sorted(scores_and_boxes, key=lambda x: -x[0])
        scores, boxes = zip(*scores_and_boxes)

        return scores, boxes


    def get_batch_faceboxes(self, image_list:List[np.ndarray], *, batch_size=-1, threshold=0.2)->List[Tuple[List[float],List[np.ndarray]]]:
        """
        image_list: List[np.ndarray] 同じサイズの画像のリスト
        batch_size: image_listの中から何個処理するかを指定する. デフォルトの場合はgpuの空きメモリから算出する
        threshold: faceboxesのconfの閾値

        return: List[Tuple[confのリスト],[BoundingBoxのリスト]]
        """

        if len(image_list) == 0:
            return [([],[])]

        im_height, im_width, im_ch = image_list[0].shape

        if batch_size == -1:
            # batch_sizeが未指定の場合はgpuの空きメモリと画像1枚あたりの容量から許容枚数を算出し、安全マージンの7掛けした値をbatch_sizeとして使用する
            torch.cuda.empty_cache()
            batch_size = get_faceboxes_max_batch_size(width=im_width, height=im_height, ch=im_ch)
            batch_size = int(batch_size * 0.7)

        resize = 1

        results = []

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        for i in range(0, int(len(image_list) / batch_size) + 1):
            images = image_list[batch_size * i: min(len(image_list), batch_size * (i+1))]

            im_height, im_width, im_ch = images[0].shape
            scale = torch.Tensor([im_width, im_height, im_width, im_height])
            images = np.array(images).reshape(len(images), im_height, im_width, im_ch)

            imgs = torch.from_numpy(images)
            imgs = imgs.to(self.device)
            scale = scale.to(self.device)

            imgs = imgs.to(torch.float32)
            imgs = torch.sub(imgs, torch.tensor((104,117,123)).to(self.device))
            imgs = imgs.permute(0,3,1,2)

            loc, conf = self.net(imgs)

            boxes = batch_decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize

            boxes = boxes.cpu().numpy()
            scores = conf.data.cpu().numpy()[:, :, 1]

            thres_tuple = torch.nonzero(torch.where(conf[:,:,1] > threshold, 1, 0), as_tuple=True)
            image_inds = thres_tuple[0].data.cpu().numpy()
            boxes_inds = thres_tuple[1].data.cpu().numpy()

            # ここgpu化しなくても処理時間が支配的ではないのでcpuでやる
            result = []
            for i in range(0, len(images)):
                inds = boxes_inds[np.where(image_inds == i)]
                if len(inds) == 0:
                    result.append(([],[]))
                    continue

                scores_and_boxes = zip(scores[i][inds], boxes[i][inds])
                scores_and_boxes = sorted(scores_and_boxes, key=lambda x: -x[0])
                s, b = zip(*scores_and_boxes)
                result.append((s,b))

            results += result

            # 以降gpu上の画像は使わないので解放する
            # 速度に影響なさそうなのでループごとに実行
            del imgs
            torch.cuda.empty_cache()

        return results