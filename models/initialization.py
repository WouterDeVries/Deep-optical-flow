import torch
import torch.nn as nn
import torch.nn.functional as F
from .FE import BasicConv2d
import pdb
from .submodules import BuildVolume4d
from .submodules import ArgminVolume4d
from .submodules import DistributedVolumeArgmin
from utils.visualization import flow_to_color
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random


class INIT(nn.Module):
    """
    Tile hypothesis initialization
    input: dual feature pyramid
    output: initial tile hypothesis pyramid
    """
    def __init__(self, args):
        super().__init__()
        self.maxfx = args.maxfx
        self.maxfy = args.maxfy
        self.init_flow_gt = args.init_flow_gt
        self.init_slant_gt = args.init_slant_gt
        fea_c1x = args.fea_c[4]
        fea_c2x = args.fea_c[3]
        fea_c4x = args.fea_c[2]
        fea_c8x = args.fea_c[1]
        fea_c16x = args.fea_c[0]
        self.tile_conv1x = nn.Sequential(
            BasicConv2d(fea_c1x, fea_c1x, 4, 4, 0, 1),
            nn.Conv2d(fea_c1x, fea_c1x, 1, 1, 0, bias=False)
        )

        self.tile_conv2x = nn.Sequential(
            BasicConv2d(fea_c2x, fea_c2x, 4, 4, 0, 1),
            nn.Conv2d(fea_c2x, fea_c2x, 1, 1, 0, bias=False)
        )

        self.tile_conv4x = nn.Sequential(
            BasicConv2d(fea_c4x, fea_c4x, 4, 4, 0, 1),
            nn.Conv2d(fea_c4x, fea_c4x, 1, 1, 0, bias=False)
        )

        self.tile_conv8x = nn.Sequential(
            BasicConv2d(fea_c8x, fea_c8x, 4, 4, 0, 1),
            nn.Conv2d(fea_c8x, fea_c8x, 1, 1, 0, bias=False)
        )

        self.tile_conv16x = nn.Sequential(
            BasicConv2d(fea_c16x, fea_c16x, 4, 4, 0, 1),
            nn.Conv2d(fea_c16x, fea_c16x, 1, 1, 0, bias=False)
        )

        self.tile_fea_dscrpt16x = BasicConv2d(fea_c16x+1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt8x = BasicConv2d(fea_c8x+1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt4x = BasicConv2d(fea_c4x+1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt2x = BasicConv2d(fea_c2x+1, 13, 1, 1, 0, 1)
        self.tile_fea_dscrpt1x = BasicConv2d(fea_c1x+1, 13, 1, 1, 0, 1)

        self._build_volume_4d = BuildVolume4d()
        self._argmin_volume_4d = ArgminVolume4d()
        self._volume_argmin = DistributedVolumeArgmin()

        self.img_no = 1


    def tile_features(self, fea_l, fea_r):
        right_fea_pad = [0, 3, 0, 3]

        tile_fea_l1x = self.tile_conv1x(fea_l[-1])
        padded_fea_r1x = F.pad(fea_r[-1], right_fea_pad)
        self.tile_conv1x[0][0].stride = (4, 4)
        tile_fea_r1x = self.tile_conv1x(padded_fea_r1x)
        self.tile_conv1x[0][0].stride = (4, 4)

        tile_fea_l2x = self.tile_conv2x(fea_l[-2])
        padded_fea_r2x = F.pad(fea_r[-2], right_fea_pad)
        self.tile_conv2x[0][0].stride = (3, 3)
        tile_fea_r2x = self.tile_conv2x(padded_fea_r2x)
        self.tile_conv2x[0][0].stride = (4, 4)

        tile_fea_l4x = self.tile_conv4x(fea_l[-3])
        padded_fea_r4x = F.pad(fea_r[-3], right_fea_pad)
        self.tile_conv4x[0][0].stride = (1, 1)
        tile_fea_r4x = self.tile_conv4x(padded_fea_r4x)
        self.tile_conv4x[0][0].stride = (4, 4)

        tile_fea_l8x = self.tile_conv8x(fea_l[-4])
        padded_fea_r8x = F.pad(fea_r[-4], right_fea_pad)
        self.tile_conv8x[0][0].stride = (1, 1)
        tile_fea_r8x = self.tile_conv8x(padded_fea_r8x)
        self.tile_conv8x[0][0].stride = (4, 4)

        tile_fea_l16x = self.tile_conv16x(fea_l[-5])
        padded_fea_r16x = F.pad(fea_r[-5], right_fea_pad)
        self.tile_conv16x[0][0].stride = (1, 1)
        tile_fea_r16x = self.tile_conv16x(padded_fea_r16x)
        self.tile_conv16x[0][0].stride = (4, 4)

        return [
            [tile_fea_l16x, tile_fea_r16x],
            [tile_fea_l8x, tile_fea_r8x],
            [tile_fea_l4x, tile_fea_r4x],
            [tile_fea_l2x, tile_fea_r2x],
            [tile_fea_l1x, tile_fea_r1x],
        ]


    def tile_hypothesis_pyramid(self, tile_feature_pyramid, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt):
        # init_tile_cost16x = self._build_volume_4d(tile_feature_pyramid[0][0], tile_feature_pyramid[0][1])
        # init_tile_cost8x = self._build_volume_4d(tile_feature_pyramid[1][0], tile_feature_pyramid[1][1])
        # init_tile_cost4x = self._build_volume_4d(tile_feature_pyramid[2][0], tile_feature_pyramid[2][1])
        # init_tile_cost2x = self._build_volume_4d(tile_feature_pyramid[3][0], tile_feature_pyramid[3][1])
        # init_tile_cost1x = self._build_volume_4d(tile_feature_pyramid[4][0], tile_feature_pyramid[4][1])

        # init_tile_flow16x, min_tile_cost16x = self._argmin_volume_4d(init_tile_cost16x, 0.25, 0.25)
        # init_tile_flow8x, min_tile_cost8x = self._argmin_volume_4d(init_tile_cost8x, 0.25, 0.25)
        # init_tile_flow4x, min_tile_cost4x = self._argmin_volume_4d(init_tile_cost4x, 0.25, 0.25)
        # init_tile_flow2x, min_tile_cost2x = self._argmin_volume_4d(init_tile_cost2x, 0.75, 0.75)
        # init_tile_flow1x, min_tile_cost1x = self._argmin_volume_4d(init_tile_cost1x, 0.75, 0.75)

        init_tile_flow16x, min_tile_cost16x = self._volume_argmin(tile_feature_pyramid[0][0], tile_feature_pyramid[0][1], 0.25, 0.25)
        init_tile_flow8x, min_tile_cost8x = self._volume_argmin(tile_feature_pyramid[1][0], tile_feature_pyramid[1][1], 0.25, 0.25)
        init_tile_flow4x, min_tile_cost4x = self._volume_argmin(tile_feature_pyramid[2][0], tile_feature_pyramid[2][1], 0.25, 0.25)
        init_tile_flow2x, min_tile_cost2x = self._volume_argmin(tile_feature_pyramid[3][0], tile_feature_pyramid[3][1], 0.75, 0.75)
        init_tile_flow1x, min_tile_cost1x = self._volume_argmin(tile_feature_pyramid[4][0], tile_feature_pyramid[4][1], 1.0, 1.0)

        # Only if initialization loss is not used
        init_tile_cost16x = torch.zeros(1,1,1,1,1).cuda()
        init_tile_cost8x = torch.zeros(1,1,1,1,1).cuda()
        init_tile_cost4x = torch.zeros(1,1,1,1,1).cuda()
        init_tile_cost2x = torch.zeros(1,1,1,1,1).cuda()
        init_tile_cost1x = torch.zeros(1,1,1,1,1).cuda()

        min_tile_cost16x = torch.unsqueeze(min_tile_cost16x, 1)
        min_tile_cost8x = torch.unsqueeze(min_tile_cost8x, 1)
        min_tile_cost4x = torch.unsqueeze(min_tile_cost4x, 1)
        min_tile_cost2x = torch.unsqueeze(min_tile_cost2x, 1)
        min_tile_cost1x = torch.unsqueeze(min_tile_cost1x, 1)

        tile_dscrpt16x = self.tile_fea_dscrpt16x(torch.cat([min_tile_cost16x, tile_feature_pyramid[0][0]], 1))
        tile_dscrpt8x = self.tile_fea_dscrpt8x(torch.cat([min_tile_cost8x, tile_feature_pyramid[1][0]], 1))
        tile_dscrpt4x = self.tile_fea_dscrpt4x(torch.cat([min_tile_cost4x, tile_feature_pyramid[2][0]], 1))
        tile_dscrpt2x = self.tile_fea_dscrpt2x(torch.cat([min_tile_cost2x, tile_feature_pyramid[3][0]], 1))
        tile_dscrpt1x = self.tile_fea_dscrpt1x(torch.cat([min_tile_cost1x, tile_feature_pyramid[4][0]], 1))

        if self.init_flow_gt or self.init_slant_gt:
            _, H16x, W16x, _ = init_tile_flow16x.shape
            _, H8x, W8x, _ = init_tile_flow8x.shape
            _, H4x, W4x, _ = init_tile_flow4x.shape
            _, H2x, W2x, _ = init_tile_flow2x.shape
            _, H1x, W1x, _ = init_tile_flow1x.shape

            transform16x = transforms.Resize((H16x, W16x))
            transform8x = transforms.Resize((H8x, W8x))
            transform4x = transforms.Resize((H4x, W4x))
            transform2x = transforms.Resize((H2x, W2x))
            transform1x = transforms.Resize((H1x, W1x))

        if self.init_flow_gt:
            fx16x = transform16x(flow_gt[:,:,:,0])
            fx8x = transform8x(flow_gt[:,:,:,0])
            fx4x = transform4x(flow_gt[:,:,:,0])
            fx2x = transform2x(flow_gt[:,:,:,0])
            fx1x = transform1x(flow_gt[:,:,:,0])

            fy16x = transform16x(flow_gt[:,:,:,1])
            fy8x = transform8x(flow_gt[:,:,:,1])
            fy4x = transform4x(flow_gt[:,:,:,1])
            fy2x = transform2x(flow_gt[:,:,:,1])
            fy1x = transform1x(flow_gt[:,:,:,1])

            init_tile_flow16x = torch.cat((fx16x.unsqueeze(3), fy16x.unsqueeze(3)), 3)
            init_tile_flow8x = torch.cat((fx8x.unsqueeze(3), fy8x.unsqueeze(3)), 3)
            init_tile_flow4x = torch.cat((fx4x.unsqueeze(3), fy4x.unsqueeze(3)), 3)
            init_tile_flow2x = torch.cat((fx2x.unsqueeze(3), fy2x.unsqueeze(3)), 3)
            init_tile_flow1x = torch.cat((fx1x.unsqueeze(3), fy1x.unsqueeze(3)), 3)

        else:
            fx16x = init_tile_flow16x[:,:,:,0]
            fx8x = init_tile_flow8x[:,:,:,0]
            fx4x = init_tile_flow4x[:,:,:,0]
            fx2x = init_tile_flow2x[:,:,:,0]
            fx1x = init_tile_flow1x[:,:,:,0]

            fy16x = init_tile_flow16x[:,:,:,1]
            fy8x = init_tile_flow8x[:,:,:,1]
            fy4x = init_tile_flow4x[:,:,:,1]
            fy2x = init_tile_flow2x[:,:,:,1]
            fy1x = init_tile_flow1x[:,:,:,1]

        fx16x = fx16x.float().unsqueeze(1)
        fx8x = fx8x.float().unsqueeze(1)
        fx4x = fx4x.float().unsqueeze(1)
        fx2x = fx2x.float().unsqueeze(1)
        fx1x = fx1x.float().unsqueeze(1)

        fy16x = fy16x.float().unsqueeze(1)
        fy8x = fy8x.float().unsqueeze(1)
        fy4x = fy4x.float().unsqueeze(1)
        fy2x = fy2x.float().unsqueeze(1)
        fy1x = fy1x.float().unsqueeze(1)

        if self.init_slant_gt:
            fxx16x = transform16x(fxx_gt)
            fxx8x = transform8x(fxx_gt)
            fxx4x = transform4x(fxx_gt)
            fxx2x = transform2x(fxx_gt)
            fxx1x = transform1x(fxx_gt)

            fxy16x = transform16x(fxy_gt)
            fxy8x = transform8x(fxy_gt)
            fxy4x = transform4x(fxy_gt)
            fxy2x = transform2x(fxy_gt)
            fxy1x = transform1x(fxy_gt)

            fyx16x = transform16x(fyx_gt)
            fyx8x = transform8x(fyx_gt)
            fyx4x = transform4x(fyx_gt)
            fyx2x = transform2x(fyx_gt)
            fyx1x = transform1x(fyx_gt)

            fyy16x = transform16x(fyy_gt)
            fyy8x = transform8x(fyy_gt)
            fyy4x = transform4x(fyy_gt)
            fyy2x = transform2x(fyy_gt)
            fyy1x = transform1x(fyy_gt)

        else:
            fxx16x = torch.zeros_like(fx16x)
            fxx8x = torch.zeros_like(fx8x)
            fxx4x = torch.zeros_like(fx4x)
            fxx2x = torch.zeros_like(fx2x)
            fxx1x = torch.zeros_like(fx1x)

            fxy16x = torch.zeros_like(fx16x)
            fxy8x = torch.zeros_like(fx8x)
            fxy4x = torch.zeros_like(fx4x)
            fxy2x = torch.zeros_like(fx2x)
            fxy1x = torch.zeros_like(fx1x)

            fyx16x = torch.zeros_like(fy16x)
            fyx8x = torch.zeros_like(fy8x)
            fyx4x = torch.zeros_like(fy4x)
            fyx2x = torch.zeros_like(fy2x)
            fyx1x = torch.zeros_like(fy1x)

            fyy16x = torch.zeros_like(fy16x)
            fyy8x = torch.zeros_like(fy8x)
            fyy4x = torch.zeros_like(fy4x)
            fyy2x = torch.zeros_like(fy2x)
            fyy1x = torch.zeros_like(fy1x)

        tile_hyp16x = torch.cat([fx16x, fxx16x, fxy16x, tile_dscrpt16x, fy16x, fyx16x, fyy16x], 1)
        tile_hyp8x = torch.cat([fx8x, fxx8x, fxy8x, tile_dscrpt8x, fy8x, fyx8x, fyy8x], 1)
        tile_hyp4x = torch.cat([fx4x, fxx4x, fxy4x, tile_dscrpt4x, fy4x, fyx4x, fyy4x], 1)
        tile_hyp2x = torch.cat([fx2x, fxx2x, fxy2x, tile_dscrpt2x, fy2x, fyx2x, fyy2x], 1)
        tile_hyp1x = torch.cat([fx1x, fxx1x, fxy1x, tile_dscrpt1x, fy1x, fyx1x, fyy1x], 1)

        # flow_img16x = flow_to_color(init_tile_flow16x[0,:,:].cpu().numpy())
        # flow_img8x = flow_to_color(init_tile_flow8x[0,:,:].cpu().numpy())
        # flow_img4x = flow_to_color(init_tile_flow4x[0,:,:].cpu().numpy())
        # flow_img2x = flow_to_color(init_tile_flow2x[0,:,:].cpu().numpy())
        # flow_img1x = flow_to_color(init_tile_flow1x[0,:,:].cpu().numpy())

        # # self.img_no = random.randint(0,1000)
        # filename_string = "img/current_model/(" + str(self.img_no) + ") "

        # plt.imsave(filename_string + "flow_init16x.png", flow_img16x)
        # plt.imsave(filename_string + "flow_init8x.png", flow_img8x)
        # plt.imsave(filename_string + "flow_init4x.png", flow_img4x)
        # plt.imsave(filename_string + "flow_init2x.png", flow_img2x)
        # plt.imsave(filename_string + "flow_init1x.png", flow_img1x)

        # self.img_no = self.img_no + 1

        return [
            [
                init_tile_cost16x,
                init_tile_cost8x,
                init_tile_cost4x,
                init_tile_cost2x,
                init_tile_cost1x,
            ],
            [
                tile_hyp16x,
                tile_hyp8x,
                tile_hyp4x,
                tile_hyp2x,
                tile_hyp1x,
            ]
        ]


    def forward(self, fea_l_pyramid, fea_r_pyramid, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt):
        tile_feature_duo_pyramid = self.tile_features(fea_l_pyramid, fea_r_pyramid)
        init_cv_pyramid, init_hypo_pyramid = self.tile_hypothesis_pyramid(tile_feature_duo_pyramid, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt)

        return [init_cv_pyramid, init_hypo_pyramid]
