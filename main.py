from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models.HITNet import HITNet
from loss.total_loss import global_loss
from loss.propagation_loss import prop_loss
from utils import *
from torch.utils.data import DataLoader
import gc
import json
from datetime import datetime
from utils.saver import Saver
import pdb
import warnings
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import torch.autograd.profiler as profiler

cudnn.benchmark = True
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='HITNet')
parser.add_argument('--maxfx', type=int, default=512, help='maximum flow in x (one-directional)')
parser.add_argument('--maxfy', type=int, default=512, help='maximum flow in y (one-directional)')
parser.add_argument('--fea_c', type=int, nargs='+', default=[32, 24, 24, 16, 16], help='feature extraction channels')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.0004, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate (epoch, epoch: rate, rate)')
parser.add_argument('--ckpt_start_epoch', type=int, default=0, help='the epochs at which the program start saving ckpt')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

parser.add_argument('--resume', type=str, help='continue training the model')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--loadfeat', help='load feature extraction weights and freeze them')
parser.add_argument('--test_img', help='the directory to save testing images, skips training')
parser.add_argument('--init_flow_gt', action='store_true', help='initialize with flow ground truth')
parser.add_argument('--init_slant_gt', action='store_true', help='initialize with slant ground truth')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)
if args.test_img:
    os.makedirs(args.test_img, exist_ok=True)

# set device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create summary logger
print()
saver = Saver(args)
print("creating new summary file")
logger = SummaryWriter(saver.experiment_dir)

# create log text file
logfilename = saver.experiment_dir + '/log.txt'
with open(logfilename, 'a') as log:
    log.write('-------------------NEW RUN-------------------\n')
    log.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log.write('\n')
    json.dump(args.__dict__, log, indent=2)
    log.write('\n')

# dataset, dataloader
flow_dataset = __datasets__[args.dataset]
train_dataset = flow_dataset(args.datapath, args.trainlist, training=True)
test_dataset = flow_dataset(args.datapath, args.testlist, training=False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=0)

# model, optimizer
model = HITNet(args)
model = nn.DataParallel(model)
model.to(device)

# freeze feature extraction weights
if args.loadfeat:
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "feature_extractor" in name or "tile_conv" in name:
                param.requires_grad = False

# optimizer, filter out feature extraction from parameters
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# load parameters
start_epoch = 0
if args.resume:
    print("resume training with model: {}".format(args.resume))
    loaded_state_dict = torch.load(args.resume)
    model.load_state_dict(loaded_state_dict['model'])
    optimizer.load_state_dict(loaded_state_dict['optimizer'])
    start_epoch = loaded_state_dict['epoch'] + 1
elif args.loadckpt:
    print("loading model: {}".format(args.loadckpt))
    loaded_state_dict = torch.load(args.loadckpt)["model"]
    current_state_dict = model.state_dict()
    # only load parameters whose size still matches current parameter size
    new_state_dict = {k:v if v.size() == current_state_dict[k].size() else current_state_dict[k] for k,v in zip(current_state_dict.keys(), loaded_state_dict.values())}
    model.load_state_dict(new_state_dict, strict=False)
elif args.loadfeat:
    print("loading feature extraction weights from model: {}".format(args.loadfeat))
    loaded_state_dict = torch.load(args.loadfeat)["model"]
    current_state_dict = model.state_dict()
    new_state_dict = {k:v if "feature_extractor" in k or "tile_conv" in k else current_state_dict[k] for k,v in zip(current_state_dict.keys(), loaded_state_dict.values())}
    model.load_state_dict(new_state_dict)
print("start at epoch {}".format(start_epoch))


def train():
    min_EPE = 1000

    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        print()

        # training
        avg_train_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            # if batch_idx == 0:
            #     break
            start_time = time.time()
            loss, scalar_outputs = train_sample(sample)
            avg_train_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {}, time = {:.3f}'.format(epoch_idx + 1, args.epochs,
                                                                                       batch_idx + 1,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
            with open(logfilename, 'a') as log:
                log.write('Epoch {}/{}, Iter {}/{}, train loss = {}, time = {:.3f}\n'.format(epoch_idx + 1, args.epochs,
                                                                                                 batch_idx + 1,
                                                                                                 len(TrainImgLoader), loss,
                                                                                                 time.time() - start_time))
        avg_train_scalars = avg_train_scalars.mean()
        save_scalars(logger, 'train', avg_train_scalars, epoch_idx + 1)
        print("\n", "avg_train_scalars", avg_train_scalars, "\n")
        with open(logfilename, 'a') as log:
            log.write('\navg_train_scalars ')
            js = json.dumps(avg_train_scalars)
            log.write(js)
            log.write('\n\n')
        if (epoch_idx + 1) % args.save_freq == 0 and epoch_idx >= args.ckpt_start_epoch:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(saver.experiment_dir, epoch_idx + 1))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        img_no = 1
        for batch_idx, sample in enumerate(TestImgLoader):
            # if batch_idx == 20:
            #     break
            global_step = len(TestImgLoader) * epoch_idx + batch_idx + 1
            start_time = time.time()
            loss, scalar_outputs = test_sample(sample, img_no)
            img_no = img_no + 1
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {}, time = {:3f}'.format(epoch_idx + 1, args.epochs,
                                                                                 batch_idx + 1,
                                                                                 len(TestImgLoader), loss,
                                                                                 time.time() - start_time))
            with open(logfilename, 'a') as log:
                log.write('Epoch {}/{}, Iter {}/{}, test loss = {}, time = {:.3f}\n'.format(epoch_idx + 1, args.epochs,
                                                                                            batch_idx + 1,
                                                                                            len(TestImgLoader), loss,
                                                                                            time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        if avg_test_scalars['EPE'][-1] < min_EPE:
            min_EPE = avg_test_scalars['EPE'][-1]
            minEPE_epoch = epoch_idx
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/bestEPE_checkpoint.ckpt".format(saver.experiment_dir))
        save_scalars(logger, 'test', avg_test_scalars, epoch_idx + 1)
        print("\n", "avg_test_scalars", avg_test_scalars, "\n")
        with open(logfilename, 'a') as log:
            log.write('\naverage_test_scalars ')
            js = json.dumps(avg_test_scalars)
            log.write(js)
            log.write('\n\n')
        with open(args.test_img + "final_eval.txt", 'a') as log:
            log.write('\naverage_test_scalars ')
            js = json.dumps(avg_test_scalars)
            log.write(js)
            log.write('\n\n')
        gc.collect()

    with open(logfilename, 'a') as log:
        log.write('min_EPE: {}/{}'.format(min_EPE, minEPE_epoch))


# train one sample
def train_sample(sample):
    model.train()

    img1, img2, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt = sample['img1'], sample['img2'], sample['flow'], sample['fxx_gt'], sample['fxy_gt'], sample['fyx_gt'], sample['fyy_gt']

    img1 = img1.to(device)
    img2 = img2.to(device)
    fx_gt = flow_gt[:,:,:,0].to(device).unsqueeze(1)
    fy_gt = flow_gt[:,:,:,1].to(device).unsqueeze(1)
    fxx_gt = fxx_gt.to(device).unsqueeze(1)
    fxy_gt = fxy_gt.to(device).unsqueeze(1)
    fyx_gt = fyx_gt.to(device).unsqueeze(1)
    fyy_gt = fyy_gt.to(device).unsqueeze(1)

    optimizer.zero_grad()

    outputs = model(img1, img2, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt)
    init_cv_pyramid = outputs["init_cv_pyramid"]
    fx_pyramid = outputs["fx_pyramid"]
    fxx_pyramid = outputs["fxx_pyramid"]
    fxy_pyramid = outputs["fxy_pyramid"]
    fy_pyramid = outputs["fy_pyramid"]
    fyx_pyramid = outputs["fyx_pyramid"]
    fyy_pyramid = outputs["fyy_pyramid"]
    w_pyramid = outputs["w_pyramid"]

    prop_diff_pyramid = []
    for fx, fy in zip(fx_pyramid, fy_pyramid):
        fx_est_single = fx[0,:,:,:].squeeze(0)
        fy_est_single = fy[0,:,:,:].squeeze(0)
        flow_est_single = torch.cat((fx_est_single.unsqueeze(2), fy_est_single.unsqueeze(2)), 2)
        flow_gt_single = flow_gt.squeeze(0)

        H, W, F = flow_est_single.shape
        flow_diff = torch.norm(flow_gt_single.to(device) - flow_est_single, p=2, dim=2)
        flow_diff = flow_diff.view(H, W).unsqueeze(0).unsqueeze(0)
        prop_diff_pyramid.append(flow_diff)

    loss_fx = global_loss(init_cv_pyramid, fx_pyramid, fxx_pyramid, fxy_pyramid, w_pyramid,
                       fx_gt, fxx_gt, fxy_gt, args.maxfx, prop_diff_pyramid,
                       lambda_init=0, lambda_prop=1, lambda_slant=1, lambda_w=1)
    loss_fx_mean = torch.mean(torch.cat([loss_fx[0], loss_fx[1], loss_fx[2]], dim=0))
    # loss_fx_mean = torch.mean(torch.cat([loss_fx[0], loss_fx[2]], dim=0))

    loss_fy = global_loss(init_cv_pyramid, fy_pyramid, fyx_pyramid, fyy_pyramid, w_pyramid,
                       fy_gt, fyx_gt, fyy_gt, args.maxfy, prop_diff_pyramid,
                       lambda_init=0, lambda_prop=1, lambda_slant=1, lambda_w=1)
    loss_fy_mean = torch.mean(torch.cat([loss_fy[0], loss_fy[1], loss_fy[2]], dim=0))
    # loss_fy_mean = torch.mean(torch.cat([loss_fy[0], loss_fy[2]], dim=0))

    loss = loss_fx_mean + loss_fy_mean
    scalar_outputs = {"total_loss": loss}

    scalar_outputs["loss_fx_prop"] = torch.mean(loss_fx[0])
    scalar_outputs["loss_fy_prop"] = torch.mean(loss_fy[0])
    scalar_outputs["loss_prop"] = scalar_outputs["loss_fx_prop"] + scalar_outputs["loss_fy_prop"]
    scalar_outputs["loss_fx_slant"] = torch.mean(loss_fx[1])
    scalar_outputs["loss_fy_slant"] = torch.mean(loss_fy[1])
    scalar_outputs["loss_slant"] = scalar_outputs["loss_fx_slant"] + scalar_outputs["loss_fy_slant"]
    scalar_outputs["loss_fx_w"] = torch.mean(loss_fx[2])
    scalar_outputs["loss_fy_w"] = torch.mean(loss_fy[2])
    scalar_outputs["loss_w"] = scalar_outputs["loss_fx_w"] + scalar_outputs["loss_fy_w"]

    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)


# test one sample
@make_nograd_func
def test_sample(sample, img_no):
    model.eval()

    img1, img2, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt = sample['img1'], sample['img2'], sample['flow'], sample['fxx_gt'], sample['fxy_gt'], sample['fyx_gt'], sample['fyy_gt']

    img1 = img1.to(device)
    img2 = img2.to(device)
    fx_gt = flow_gt[:,:,:,0].to(device).unsqueeze(1)
    fy_gt = flow_gt[:,:,:,1].to(device).unsqueeze(1)
    fxx_gt = fxx_gt.to(device).unsqueeze(1)
    fxy_gt = fxy_gt.to(device).unsqueeze(1)
    fyx_gt = fyx_gt.to(device).unsqueeze(1)
    fyy_gt = fyy_gt.to(device).unsqueeze(1)

    outputs = model(img1, img2, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt)

    init_cv_pyramid = outputs["init_cv_pyramid"]
    fx_pyramid = outputs["fx_pyramid"]
    fxx_pyramid = outputs["fxx_pyramid"]
    fxy_pyramid = outputs["fxy_pyramid"]
    fy_pyramid = outputs["fy_pyramid"]
    fyx_pyramid = outputs["fyx_pyramid"]
    fyy_pyramid = outputs["fyy_pyramid"]
    w_pyramid = outputs["w_pyramid"]

    prop_diff_pyramid = []
    for fx, fy in zip(fx_pyramid, fy_pyramid):
        fx_est_single = fx[0,:,:,:].squeeze(0)
        fy_est_single = fy[0,:,:,:].squeeze(0)
        flow_est_single = torch.cat((fx_est_single.unsqueeze(2), fy_est_single.unsqueeze(2)), 2)
        flow_gt_single = flow_gt.squeeze(0)

        H, W, F = flow_est_single.shape
        flow_diff = torch.norm(flow_gt_single.to(device) - flow_est_single, p=2, dim=2)
        flow_diff = flow_diff.view(H, W).unsqueeze(0).unsqueeze(0)
        prop_diff_pyramid.append(flow_diff)

    loss_fx = global_loss(init_cv_pyramid, fx_pyramid, fxx_pyramid, fxy_pyramid, w_pyramid,
                       fx_gt, fxx_gt, fxy_gt, args.maxfx, prop_diff_pyramid,
                       lambda_init=1, lambda_prop=1, lambda_slant=1, lambda_w=1)
    loss_fx_mean = torch.mean(torch.cat([loss_fx[0], loss_fx[1], loss_fx[2]], dim=0))
    # loss_fx_mean = torch.mean(torch.cat([loss_fx[0], loss_fx[2]], dim=0))

    loss_fy = global_loss(init_cv_pyramid, fy_pyramid, fyx_pyramid, fyy_pyramid, w_pyramid,
                       fy_gt, fyx_gt, fyy_gt, args.maxfy, prop_diff_pyramid,
                       lambda_init=1, lambda_prop=1, lambda_slant=1, lambda_w=1)
    loss_fy_mean = torch.mean(torch.cat([loss_fy[0], loss_fy[1], loss_fy[2]], dim=0))
    # loss_fy_mean = torch.mean(torch.cat([loss_fy[0], loss_fy[2]], dim=0))

    loss = loss_fx_mean + loss_fy_mean
    scalar_outputs = {"total_loss": loss}

    scalar_outputs["loss_fx_prop"] = torch.mean(loss_fx[0])
    scalar_outputs["loss_fy_prop"] = torch.mean(loss_fy[0])
    scalar_outputs["loss_prop"] = scalar_outputs["loss_fx_prop"] + scalar_outputs["loss_fy_prop"]
    scalar_outputs["loss_fx_slant"] = torch.mean(loss_fx[1])
    scalar_outputs["loss_fy_slant"] = torch.mean(loss_fy[1])
    scalar_outputs["loss_slant"] = scalar_outputs["loss_fx_slant"] + scalar_outputs["loss_fy_slant"]
    scalar_outputs["loss_fx_w"] = torch.mean(loss_fx[2])
    scalar_outputs["loss_fy_w"] = torch.mean(loss_fy[2])
    scalar_outputs["loss_w"] = scalar_outputs["loss_fx_w"] + scalar_outputs["loss_fy_w"]

    # Evaluation
    # Flow in shape H*W*2
    # Slants in shape H*W
    img1_fn = sample["img1_filename"][0][30:39]
    img2_fn = sample["img2_filename"][0][30:39]
    fx_est, fy_est = outputs['final_fx'], outputs['final_fy']

    fx_est = fx_est[0,:,:,:].squeeze(0)
    fy_est = fy_est[0,:,:,:].squeeze(0)
    flow_est = torch.cat((fx_est.unsqueeze(2), fy_est.unsqueeze(2)), 2)
    fxx_est = fxx_pyramid[10][0,:,:,:].squeeze(0).cpu().detach().numpy()
    fxy_est = fxy_pyramid[10][0,:,:,:].squeeze(0).cpu().detach().numpy()
    fyx_est = fyx_pyramid[10][0,:,:,:].squeeze(0).cpu().detach().numpy()
    fyy_est = fyy_pyramid[10][0,:,:,:].squeeze(0).cpu().detach().numpy()

    flow_gt = flow_gt[0,:,:,:]
    fx_gt = flow_gt[:,:,0]
    fy_gt = flow_gt[:,:,1]
    fxx_gt = fxx_gt.squeeze(0).squeeze(0).cpu().numpy()
    fxy_gt = fxy_gt.squeeze(0).squeeze(0).cpu().numpy()
    fyx_gt = fyx_gt.squeeze(0).squeeze(0).cpu().numpy()
    fyy_gt = fyy_gt.squeeze(0).squeeze(0).cpu().numpy()

    mask_flow = (flow_gt < 512) & (flow_gt > -512) & (flow_gt != 0)
    mask_flow_1D = (fx_gt < 512) & (fx_gt > -512) & (fx_gt != 0) & (fy_gt < 512) & (fy_gt > -512) & (fy_gt != 0)

    # EPE
    H, W, F = flow_est.shape
    flow_diff = torch.norm(flow_gt.to(device) - flow_est, p=2, dim=2)
    flow_diff = flow_diff.view(H, W)
    EPE = flow_diff[mask_flow_1D].mean()
    EPE_str = " " + str(np.around(EPE.cpu().numpy(), 1)) + " "
    scalar_outputs["EPE"] = [torch.tensor(EPE)]

    # Visuals
    if args.test_img:
        eval_path = args.test_img + "(" + str(img_no) + ")" + " "

        flow_est_img = flow_to_color(flow_est.cpu().numpy())
        plt.imsave(eval_path + img1_fn + "_flow_est" + EPE_str + ".png", flow_est_img)
        flow_gt_img = flow_to_color(flow_gt.cpu().numpy())
        plt.imsave(eval_path + img1_fn + "_flow_gt" + EPE_str + ".png", flow_gt_img)

        flow_est_masked = flow_est * mask_flow.to(device)
        flow_est_masked_img = flow_to_color(flow_est_masked.cpu().numpy())
        plt.imsave(eval_path + img1_fn + "_flow_est_masked" + EPE_str + ".png", flow_est_masked_img)

        slant_to_color(fxx_gt, fxy_gt, eval_path + img1_fn + "_slant_fx_gt" + EPE_str + ".png")
        slant_to_color(fyx_gt, fyy_gt, eval_path + img1_fn + "_slant_fy_gt" + EPE_str + ".png")
        slant_to_color(fxx_est, fxy_est, eval_path + img1_fn + "_slant_fx_est" + EPE_str + ".png")
        slant_to_color(fyx_est, fyy_est, eval_path + img1_fn + "_slant_fy_est" + EPE_str + ".png")

    # Error threshold metrics and error image
    mask_2px = (flow_diff <= 2) * mask_flow_1D.to(device)
    mask_3px = (flow_diff <= 3) * mask_flow_1D.to(device)
    mask_4px = (flow_diff <= 4) * mask_flow_1D.to(device)
    mask_5px = (flow_diff <= 5) * mask_flow_1D.to(device)
    mask_10px = (flow_diff <= 10) * mask_flow_1D.to(device)
    mask_15px = (flow_diff <= 15) * mask_flow_1D.to(device)
    mask_20px = (flow_diff <= 20) * mask_flow_1D.to(device)
    mask_all = (flow_diff == flow_diff) * mask_flow_1D.to(device)

    error_image = np.zeros([H, W, 3], dtype=np.float32)
    error_image[mask_all.cpu().numpy()] = [255, 78, 17]
    error_image[mask_20px.cpu().numpy()] = [255, 142, 21]
    error_image[mask_15px.cpu().numpy()] = [250, 183, 51]
    error_image[mask_10px.cpu().numpy()] = [172, 179, 52]
    error_image[mask_5px.cpu().numpy()] = [105, 179, 76]
    error_image = error_image / 255.
    if args.test_img:
        plt.imsave(eval_path + img1_fn + "_error_map" + EPE_str + ".png", error_image)

    error_image2 = np.zeros([H, W, 3], dtype=np.float32)
    error_image2[mask_all.cpu().numpy()] = [255, 78, 17]
    error_image2[mask_5px.cpu().numpy()] = [255, 142, 21]
    error_image2[mask_4px.cpu().numpy()] = [250, 183, 51]
    error_image2[mask_3px.cpu().numpy()] = [172, 179, 52]
    error_image2[mask_2px.cpu().numpy()] = [105, 179, 76]
    error_image2 = error_image2 / 255.
    if args.test_img:
        plt.imsave(eval_path + img1_fn + "_error_map2" + EPE_str + ".png", error_image2)

    num_gt_points = mask_flow_1D.sum()
    error_2px = 100 - (mask_2px.sum() / num_gt_points * 100)
    error_3px = 100 - (mask_3px.sum() / num_gt_points * 100)
    error_4px = 100 - (mask_4px.sum() / num_gt_points * 100)
    error_5px = 100 - (mask_5px.sum() / num_gt_points * 100)
    error_10px = 100 - (mask_10px.sum() / num_gt_points * 100)
    error_15px = 100 - (mask_15px.sum() / num_gt_points * 100)
    error_20px = 100 - (mask_20px.sum() / num_gt_points * 100)

    error_str = str(np.around(error_2px.cpu().numpy(),2)) + "_" + str(np.around(error_3px.cpu().numpy(),2)) + "_" + str(np.around(error_4px.cpu().numpy(),2)) + "_" + str(np.around(error_5px.cpu().numpy(),2)) + "_" + str(np.around(error_10px.cpu().numpy(),2)) + "_" + str(np.around(error_15px.cpu().numpy(),2)) + "_" + str(np.around(error_20px.cpu().numpy(),2))

    scalar_outputs["2px"] = [torch.tensor(error_2px)]
    scalar_outputs["3px"] = [torch.tensor(error_3px)]
    scalar_outputs["4px"] = [torch.tensor(error_4px)]
    scalar_outputs["5px"] = [torch.tensor(error_5px)]
    scalar_outputs["10px"] = [torch.tensor(error_10px)]
    scalar_outputs["15px"] = [torch.tensor(error_15px)]
    scalar_outputs["20px"] = [torch.tensor(error_20px)]

    if args.test_img:
        save_image(img1, eval_path + img1_fn + "_normalized" + EPE_str + error_str + ".png")
        save_image(img2, eval_path + img2_fn + "_normalized" + EPE_str + ".png")
    
    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    train()
