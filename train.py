from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import math
from models.faceboxes import FaceBoxes

parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--training_dataset', default='./data/WIDER_FACE', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

img_dim = 1024
rgb_means = (104, 117, 123)
num_classes = 2
batch_size = args.batch_size
weight_decay = args.weight_decay
gamma = args.gamma
momentum = args.momentum

net = FaceBoxes('train', img_dim, num_classes)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = VOCDetection(args.training_dataset, preproc(img_dim, rgb_means), AnnotationTransform())

    epoch_size = math.ceil(len(dataset) / args.batch_size)
    max_iter = args.max_epoch * epoch_size

    stepvalues = (200 * epoch_size, 250 * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder + 'FaceBoxes_epoch_' + repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        # forward
        out = net(images)
        
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) +
              '|| Totel iter ' + repr(iteration) + ' || L: %.4f C: %.4f||' % (cfg['loc_weight']*loss_l.item(), loss_c.item()) +
              'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), args.save_folder + 'Final_FaceBoxes.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 0:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
if __name__ == '__main__':
    train()
