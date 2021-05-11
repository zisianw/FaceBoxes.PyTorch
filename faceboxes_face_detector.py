import os
import torch
import numpy as np
from .data import cfg
from .layers.functions.prior_box import PriorBox
from .models.faceboxes import FaceBoxes
from .utils.box_utils import decode

class FaceBoxesFaceDetector(object):
    def __init__(self):
        torch.set_grad_enabled(False)
        # net and model
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
        weight_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights/FaceBoxes.pth'))
        self.net = self.load_model(self.net, weight_path, True)
        self.net.eval()
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
