import argparse
import os
import shutil
import time
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datasets.factory import get_imdb
from torchvision.utils import make_grid
from custom import *
import pdb
from tensorboardX import SummaryWriter
import time
import random
import cv2

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 32)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0

def apply_maxpool(model_output):
    assert(model_output.shape[2]==model_output.shape[3]) #filter map width and height are the same. This should be true for our use case, since model input gets a square image
    pool_kernel_size = model_output.shape[2]
    pool = nn.MaxPool2d(kernel_size = pool_kernel_size, stride=1)
    return pool(model_output)

def main():
    rand_seed = 1024
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
    MODEL_SAVE_PATH = "Saved_Models/localizer_alexnet_" + str(rand_seed)
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()

    if args.vis:
        import visdom
        vis = visdom.Visdom(server='http://ec2-18-188-81-147.us-east-2.compute.amazonaws.com/',port='8097') #Change Address as per need. Can change to commandline arg
    else:
        vis = None

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = "runs/"+timestr+'/'
    writer = SummaryWriter(save_path)

    #Initializing model weights
    pretrained_alexnet = models.alexnet(pretrained=True)
    conv_layer_numbers = [0,3,6,8,10]
    for elt in conv_layer_numbers:
        #print(model.features.module[elt])
        model.features.module[elt].weight.data.copy_(pretrained_alexnet.features[elt].weight.data)
        model.features.module[elt].bias.data.copy_(pretrained_alexnet.features[elt].bias.data)

    xavier_initialized_conv_layers = [0,2,4]
    for elt in xavier_initialized_conv_layers:
        torch.nn.init.xavier_uniform_(model.classifier[elt].weight)

    print("Registering backward hook for final conv layer")
    model.classifier[4].register_backward_hook(hook)

    iter_cnt = 0
    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch count is: ",epoch,"\t","Iteration Count is: ",iter_cnt)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        iter_cnt = train(train_loader, model, criterion, optimizer, epoch, iter_cnt, writer, trainval_imdb._classes,vis)

        # evaluate on validation set
        #if epoch+1 % args.eval_freq == 0:                                        #TODO: Delete this line and uncomment the line below later
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion)
            writer.add_scalar('Validation/mAP', m1, epoch)
            writer.add_scalar('Validation/f1_score', m2, epoch)
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            if(is_best):
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    writer.close()

conv_8_out = []
def hook(module, grad_input, grad_output):
    conv_8_out.append(grad_output[0].cpu().clone())

def inv_transform(transformed_tensor):
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    return inv_normalize(transformed_tensor)

#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, iter_cnt, writer,prediction_class_list,vis=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.type(torch.FloatTensor).cuda(async=True)
        #print(input.is_cuda)
        input_var = input.cuda()
        # target_var = target
        # target_var = target

        # TODO: Get output from model
        output_var = model(input_var)
        # TODO: Perform any necessary functions on the output
        image_pred = apply_maxpool(output_var)
        # TODO: Compute loss using ``criterion``

        loss = criterion(image_pred.squeeze(), target)

        # measure metrics and record loss
        m1 = metric1(image_pred.squeeze().cpu().data, target)
        m2 = metric2(image_pred.squeeze().cpu().data, target.cpu().clone())
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1, input.size(0))
        avg_m2.update(m2, input.size(0))
        # TODO:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('Train/Loss', loss, iter_cnt)
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))
        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
        if(i==0 or i==len(train_loader)-1):   #Makes sure this visualization is twice per epoch: Once in the first batch and once in the last batch
            #Plot the first image of each batch
            writer.add_image('Train_Images_'+str(epoch)+'_'+str(iter_cnt), input[0]) 
            if(vis is not None):
                im = input[0].numpy()
                vis.image(im)
                original_image = inv_transform(input[0].clone()).cpu().detach().numpy()
                vis.image(original_image,opts={'title': "Original Image_"+str(epoch) + '_' + str(i)})


            #Getting heatmap
            EPS = 0.000001
            heatmaps = inv_transform(input[0].clone()).unsqueeze(0)
            values, indices = torch.max(target[0].clone().cpu(), 0)
            for j in range(indices.numel()):
                class_idx = indices[j].item() #Select one of the present ground truth classes
                class_name = prediction_class_list[class_idx]
                #print("GT class index is : ",class_idx)
                #activation = torch.sigmoid(output_var[0,class_idx,:,:]).clone().cpu()
                activation = output_var[0,class_idx,:,:].clone().cpu()

                heatmap = cv2.resize(activation.detach().numpy(), (input[0].shape[2], input[0].shape[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = np.swapaxes(heatmap,0,2)
                heatmap = np.swapaxes(heatmap,1,2)

                if(vis is not None):
                    #heat = heatmap.numpy()
                    text_to_display = str(epoch) + '_' + str(i) + '_' + 'heatmap_' + str(class_name)
                    #vis.text(text_to_display)
                    vis.image(heatmap,opts={'title': text_to_display})

                heatmaps = torch.cat((heatmaps, torch.from_numpy(heatmap).unsqueeze(0).float()), 0)

            img_grid = make_grid(heatmaps)
            text_to_display = str(epoch) + '_' + str(i) + '_' + 'heatmaps'
            writer.add_image(text_to_display, img_grid)

            '''
            #class_output = output_var[0][class_idx].detach().cpu()
            activations = output_var[0].cpu()
            #output_var[:,class_idx,:,:].backward()
            #gradients = model.get_activations_gradient()
            gradients = conv_8_out[-1][0][0]
            #pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            pooled_gradients =  gradients.mean(-1).mean(-1).cpu()
            for i in range(pooled_gradients.shape[0]):
                activations[i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activations, dim=0)
        heatmap = heatmap.detach()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= torch.max(heatmap)
            heatmap = heatmap.unsqueeze(0)
            #gradients = conv_8_out[-1][0][0][class_idx][0]'''
            '''conv_layer_output_value = conv_8_out[-1][0][0][class_idx][0] * class_output
            new_conv = conv_layer_output_value.unsqueeze(dim=0)'''
            ''' writer.add_image('Heatmap_'+str(epoch)+'_'+str(iter_cnt), heatmap)
            heatmap = np.mean(conv_layer_output_value, axis = -1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            if(vis is not None):
                heat = heatmap.numpy()
                vis.image(heat)'''

        iter_cnt+=1
    writer.add_scalar('mAP', avg_m1.avg, epoch)
    writer.add_scalar('f1_score', avg_m2.avg, epoch)
    return iter_cnt
        # End of train()

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input.cuda()
        target_var = target
        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        output_var = model(input_var)
        image_pred = apply_maxpool(output_var)
        loss = criterion(image_pred.squeeze(), target)
        # measure metrics and record loss
        m1 = metric1(image_pred.squeeze().cpu().data, target)
        m2 = metric2(image_pred.squeeze().cpu().data, target.cpu().clone())
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1, input.size(0))
        avg_m2.update(m2, input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))
        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))
    return avg_m1.avg, avg_m2.avg

# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_ap(gt, pred, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image. N is the batch size
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy). N is the batch size
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid].astype('float32')
        pred_cls = pred[:, cid].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP

def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    gt = target.cpu().clone().numpy()
    pred = torch.sigmoid(output)
    pred = pred.cpu().clone().numpy()
    AP = compute_ap(gt, pred)
    mAP = np.mean(AP)
    return mAP

def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    pred = torch.sigmoid(output)
    return sklearn.metrics.f1_score(target, pred > 0.5, average="samples")

if __name__ == '__main__':
    main()
