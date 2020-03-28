from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
from tensorboardX import SummaryWriter
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime

import cPickle as pkl
import network
from wsddn import WSDDN
from utils.timer import Timer

import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
import gc
import pdb
from test import test_net

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
cfg_file = 'experiments/cfgs/wsddn.yml'
pretrained_model = 'data/pretrained_model/alexnet_imagenet.npy'
output_dir = 'models/saved_model'
visualize = True
start_step = 0

vis_interval = 500
hist_interval = 2000
eval_interval = 5000
end_step = 30000

#vis_interval = 10
#hist_interval = 20
#end_step = 100
#eval_interval = 50
eval_pts = np.arange(0,end_step+1,eval_interval)
last_eval_step = eval_pts[-1]

lr_decay_steps = {150000}
lr_decay = 1. / 10

rand_seed = 1024
_DEBUG = False
use_tensorboard = True
use_visdom = True
log_grads = False

remove_all_log = False  # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config file and get hyperparameters
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load imdb and create data later
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# Create network and initialize
net = WSDDN(classes=imdb.classes, debug=_DEBUG)
print(net)
network.weights_normal_init(net, dev=0.001)
if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()
for name, param in pret_net.items():
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue

# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)

conv_layer_numbers = [0]
for elt in conv_layer_numbers:
   net.features[elt].weight.requires_grad = False
   net.features[elt].bias.requires_grad = False

'''for param in net.parameters():
        print(param.shape)
        print(param.requires_grad)     #Freeze the features part of the network
        print("==================================================================================")'''

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if use_tensorboard:
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = "runs/"+timestr+'/'
    writer = SummaryWriter(save_path)

if use_visdom:
    import visdom
    vis = visdom.Visdom(server='ec2-3-134-117-208.us-east-2.compute.amazonaws.com',port='8097')

#Create Validation Data
imdb_name_val = 'voc_2007_test'
imdb_val = get_imdb(imdb_name_val)
imdb_val.competition_mode(on=True)
save_name = '{}_{}'
thresh = 0.0001

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step + 1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    rois = blobs['rois']
    im_info = blobs['im_info']
    gt_vec = blobs['labels']
    #gt_boxes = blobs['gt_boxes']

    # forward
    net(im_data, rois, im_info, gt_vec)
    loss = net.loss
    train_loss += loss.item()
    step_cnt += 1

    # backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Log to screen
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch), lr: %.9f, momen: %.4f, wt_dec: %.6f' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps, lr,
            momentum, weight_decay)
        log_print(log_text, color='green', attrs=['bold'])
        re_cnt = True

    #TODO: evaluate the model every N iterations (N defined in handout)
    if (step) % eval_interval ==0:
        net.eval()
        aps = test_net(name = save_name, net = net, imdb = imdb_val, thresh = thresh, logger=writer, visualize=visualize, step=step)
        mAP = np.mean(aps)
        print("Average Precisions are: ",aps)
        if(step==last_eval_step):
            print("Final Step Result")
            print("Final mAP is: ",mAP)
            print("Final class-wise AP is: ")
            for id,elt in enumerate(aps):
                class_name = imdb_val._classes[id]
                print(str(class_name) + "_AP: ",aps[id])

        if visualize:
            if use_visdom:
                if step==0:
                    val_mAP_window = vis.line(X=torch.ones((1)).cpu()*step,Y=torch.ones((1)).cpu()*mAP,opts=dict(xlabel='step',ylabel='Validation_mAP',title='Validation mAP',legend=['Validation mAP']))
                else:
                    vis.line(X=torch.ones((1)).cpu()*step,Y=torch.ones((1)).cpu()*mAP,win=val_mAP_window,update='append')
            if use_tensorboard:
                for id,elt in enumerate(aps):
                    class_name = imdb_val._classes[id]
                    writer.add_scalar('Validation/AP_' + str(class_name), aps[id], step)
        net.train()

    #TODO: Perform all visualizations here
    #You can define other interval variable if you want (this is just an
    #example)
    #The intervals for different things are defined in the handout
    if visualize:
        if step % vis_interval == 0:
            print("Visualizing Loss")
            #TODO: Create required visualizations
            if use_tensorboard:
                #print('Logging to Tensorboard')
                writer.add_scalar('Train/Loss', loss, step)
            if use_visdom:
                if step==0:
                    loss_window = vis.line(X=torch.ones((1)).cpu()*step,Y=torch.Tensor([loss]).cpu(),opts=dict(xlabel='step',ylabel='Loss',title='training loss',legend=['Loss']))
                else:
                    vis.line(X=torch.ones((1)).cpu()*step,Y=torch.Tensor([loss]).cpu(),win=loss_window,update='append')
        #print(hist_interval)
        if step % hist_interval == 0:
            #Get Histograms here
            print("Getting Histogram")
            if use_tensorboard:
                state_dict = net.state_dict()
                for k,v in state_dict.items():
                    if('weight' in k):
                        writer.add_histogram('Weights_'+str(k),v,step)

                for name,param in net.named_parameters():
                    if(param.requires_grad and param.grad is not None):
                        writer.add_histogram('Gradients_' + str(name), param.grad, step)
                '''for name,param in net.named_parameters():
                    #pdb.set_trace()
                    if(param.requires_grad):
                        #pdb.set_trace()
                        writer.add_histogram(name, param.detach().cpu(), step)'''

    # Save model occasionally
    if (step % cfg.TRAIN.SNAPSHOT_ITERS == 0) and step > 0:
        save_name = os.path.join(
            output_dir, '{}_{}.h5'.format(cfg.TRAIN.SNAPSHOT_PREFIX, step))
        network.save_net(save_name, net)
        print('Saved model to {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

writer.close()
