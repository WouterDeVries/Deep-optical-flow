import torch
import torch.nn as nn
import torch.nn.functional as F
from .FE import feature_extraction_conv
from .initialization import INIT
from .tile_warping import TileWarping
from .tile_update import TileUpdate, PostTileUpdate, FinalTileUpdate, PostTileUpdateNoUp
from models.submodules import DispUpsampleBySlantedPlane, SlantDUpsampleBySlantedPlaneT4T4, SlantD2xUpsampleBySlantedPlaneT4T2
import pdb
from utils.write_pfm import write_pfm_tensor
import time

from utils.visualization import flow_to_color, slant_to_color
import matplotlib.pyplot as plt
import random


class HITNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_extractor = feature_extraction_conv(args)
        self.tile_init = INIT(args)
        self.tile_warp = TileWarping(args)
        self.tile_update0 = TileUpdate(35, 19, 32, 2, args)  # 1/16 tile refine
        self.tile_update1 = TileUpdate(70, 40, 32, 2, args)  # 1/8 tile refine
        self.tile_update2 = TileUpdate(70, 40, 32, 2, args)  # 1/4 tile refine
        self.tile_update3 = TileUpdate(70, 40, 32, 2, args)  # 1/2 tile refine
        self.tile_update4 = TileUpdate(70, 40, 32, 2, args)  # 1/1 tile refine
        self.tile_update4_1 = PostTileUpdateNoUp(43, 19, 32, 4, args)  # 1/1 tile refine
        self.tile_update5 = PostTileUpdate(35, 19, 32, 4, SlantD2xUpsampleBySlantedPlaneT4T2(), args)  # 2/1 tile refine tile_size=2
        self.tile_update6 = FinalTileUpdate(35, 2, 16, 2, DispUpsampleBySlantedPlane(2, 2), args)  # 2/1 tile refine tile_size=1

        # For training phase, we need to upsample disps using slant equation
        self.prop_disp_upsample64x = DispUpsampleBySlantedPlane(64)
        self.prop_disp_upsample32x = DispUpsampleBySlantedPlane(32)
        self.prop_disp_upsample16x = DispUpsampleBySlantedPlane(16)
        self.prop_disp_upsample8x = DispUpsampleBySlantedPlane(8)
        self.prop_disp_upsample4x = DispUpsampleBySlantedPlane(4)
        self.prop_disp_upsample2x = DispUpsampleBySlantedPlane(2, 2)
        # For training phase, we need to upsample dx and dy using nearest interpolation
        self.dxdy_upsample64x = nn.UpsamplingNearest2d(scale_factor=64)
        self.dxdy_upsample32x = nn.UpsamplingNearest2d(scale_factor=32)
        self.dxdy_upsample16x = nn.UpsamplingNearest2d(scale_factor=16)
        self.dxdy_upsample8x = nn.UpsamplingNearest2d(scale_factor=8)
        self.dxdy_upsample4x = nn.UpsamplingNearest2d(scale_factor=4)
        self.dxdy_upsample2x = nn.UpsamplingNearest2d(scale_factor=2)
        # For final disparity and each supervision signal to be positive
        # self.relu = nn.ReLU(inplace=True)
        self.img_no = 0

    def forward(self, left_img, right_img, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt):
        left_fea_pyramid = self.feature_extractor(left_img)
        right_fea_pyramid = self.feature_extractor(right_img)
        init_cv_pyramid, init_tile_pyramid = self.tile_init(left_fea_pyramid, right_fea_pyramid, flow_gt, fxx_gt, fxy_gt, fyx_gt, fyy_gt)
        refined_tile16x = self.tile_update0(left_fea_pyramid[0], right_fea_pyramid[0], init_tile_pyramid[0])[0]
        tile_update8x = self.tile_update1(left_fea_pyramid[1], right_fea_pyramid[1], init_tile_pyramid[1], refined_tile16x)
        tile_update4x = self.tile_update2(left_fea_pyramid[2], right_fea_pyramid[2], init_tile_pyramid[2], tile_update8x[0])
        tile_update2x = self.tile_update3(left_fea_pyramid[3], right_fea_pyramid[3], init_tile_pyramid[3], tile_update4x[0])
        tile_update1x = self.tile_update4(left_fea_pyramid[4], right_fea_pyramid[4], init_tile_pyramid[4], tile_update2x[0])
        refined_tile1x = self.tile_update4_1(left_fea_pyramid[2], tile_update1x[0])
        refined_tile05x = self.tile_update5(left_fea_pyramid[3], refined_tile1x)
        refined_tile025x = self.tile_update6(left_fea_pyramid[4], refined_tile05x)
        final_fx, final_fy = refined_tile025x

        # self.img_no = self.img_no + 1
        # filename_string = "img/current_model/(" + str(self.img_no) + ") "

        # flow_16x = torch.cat([refined_tile16x[:,:1,:,:].squeeze(0).squeeze(0).unsqueeze(2), refined_tile16x[:,16:17,:,:].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img16x = flow_to_color(flow_16x.cpu().numpy())
        # plt.imsave(filename_string + "11flow_16x.png", flow_img16x)
        # flow_8x = torch.cat([tile_update8x[1].squeeze(0).squeeze(0).unsqueeze(2), tile_update8x[9].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img8x = flow_to_color(flow_8x.cpu().numpy())
        # plt.imsave(filename_string + "12flow_8x.png", flow_img8x)
        # flow_4x = torch.cat([tile_update4x[1].squeeze(0).squeeze(0).unsqueeze(2), tile_update4x[9].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img4x = flow_to_color(flow_4x.cpu().numpy())
        # plt.imsave(filename_string + "13flow_4x.png", flow_img4x)
        # flow_2x = torch.cat([tile_update2x[1].squeeze(0).squeeze(0).unsqueeze(2), tile_update2x[9].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img2x = flow_to_color(flow_2x.cpu().numpy())
        # plt.imsave(filename_string + "14flow_2x.png", flow_img2x)
        # flow_1x = torch.cat([tile_update1x[1].squeeze(0).squeeze(0).unsqueeze(2), tile_update1x[9].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img1x = flow_to_color(flow_1x.cpu().numpy())
        # plt.imsave(filename_string + "15flow_1x.png", flow_img1x)
        # flow_1xref = torch.cat([refined_tile1x[:,:1,:,:].squeeze(0).squeeze(0).unsqueeze(2), refined_tile1x[:,16:17,:,:].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img1xref = flow_to_color(flow_1xref.cpu().numpy())
        # plt.imsave(filename_string + "16flow_1xref.png", flow_img1xref)
        # flow_05xref = torch.cat([refined_tile05x[:,:1,:,:].squeeze(0).squeeze(0).unsqueeze(2), refined_tile05x[:,16:17,:,:].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img05xref = flow_to_color(flow_05xref.cpu().numpy())
        # plt.imsave(filename_string + "17flow_05xref.png", flow_img05xref) 

        # slant_to_color(refined_tile16x[:,1:2,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile16x[:,2:3,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "21slantx_16x.png")
        # slant_to_color(tile_update8x[3].squeeze(0).squeeze(0).cpu().numpy(), tile_update8x[5].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "22slantx_16x.png")
        # slant_to_color(tile_update4x[3].squeeze(0).squeeze(0).cpu().numpy(), tile_update4x[5].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "23slantx_16x.png")
        # slant_to_color(tile_update2x[3].squeeze(0).squeeze(0).cpu().numpy(), tile_update2x[5].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "24slantx_16x.png")
        # slant_to_color(tile_update1x[3].squeeze(0).squeeze(0).cpu().numpy(), tile_update1x[5].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "25slantx_16x.png")
        # slant_to_color(refined_tile1x[:,1:2,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile1x[:,2:3,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "26slantx_16x.png")
        # slant_to_color(refined_tile05x[:,1:2,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile05x[:,2:3,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "27slantx_16x.png")

        # slant_to_color(refined_tile16x[:,17:18,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile16x[:,18:19,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "31slanty_16x.png")
        # slant_to_color(tile_update8x[11].squeeze(0).squeeze(0).cpu().numpy(), tile_update8x[13].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "32slanty_16x.png")
        # slant_to_color(tile_update4x[11].squeeze(0).squeeze(0).cpu().numpy(), tile_update4x[13].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "33slanty_16x.png")
        # slant_to_color(tile_update2x[11].squeeze(0).squeeze(0).cpu().numpy(), tile_update2x[13].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "34slanty_16x.png")
        # slant_to_color(tile_update1x[11].squeeze(0).squeeze(0).cpu().numpy(), tile_update1x[13].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "35slanty_16x.png")
        # slant_to_color(refined_tile1x[:,17:18,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile1x[:,18:19,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "36slanty_16x.png")
        # slant_to_color(refined_tile05x[:,17:18,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile05x[:,18:19,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "37slanty_16x.png")

        # flow_16x = torch.cat([refined_tile16x[:,:1,:,:].squeeze(0).squeeze(0).unsqueeze(2), refined_tile16x[:,16:17,:,:].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img16x = flow_to_color(flow_16x.cpu().numpy())
        # plt.imsave(filename_string + "11flow_16x.png", flow_img16x)
        # flow_8x = torch.cat([tile_update8x[2].squeeze(0).squeeze(0).unsqueeze(2), tile_update8x[10].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img8x = flow_to_color(flow_8x.cpu().numpy())
        # plt.imsave(filename_string + "12flow_8x.png", flow_img8x)
        # flow_4x = torch.cat([tile_update4x[2].squeeze(0).squeeze(0).unsqueeze(2), tile_update4x[10].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img4x = flow_to_color(flow_4x.cpu().numpy())
        # plt.imsave(filename_string + "13flow_4x.png", flow_img4x)
        # flow_2x = torch.cat([tile_update2x[2].squeeze(0).squeeze(0).unsqueeze(2), tile_update2x[10].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img2x = flow_to_color(flow_2x.cpu().numpy())
        # plt.imsave(filename_string + "14flow_2x.png", flow_img2x)
        # flow_1x = torch.cat([tile_update1x[2].squeeze(0).squeeze(0).unsqueeze(2), tile_update1x[10].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img1x = flow_to_color(flow_1x.cpu().numpy())
        # plt.imsave(filename_string + "15flow_1x.png", flow_img1x)
        # flow_1xref = torch.cat([refined_tile1x[:,:1,:,:].squeeze(0).squeeze(0).unsqueeze(2), refined_tile1x[:,16:17,:,:].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img1xref = flow_to_color(flow_1xref.cpu().numpy())
        # plt.imsave(filename_string + "16flow_1xref.png", flow_img1xref)
        # flow_05xref = torch.cat([refined_tile05x[:,:1,:,:].squeeze(0).squeeze(0).unsqueeze(2), refined_tile05x[:,16:17,:,:].squeeze(0).squeeze(0).unsqueeze(2)], dim=2)
        # flow_img05xref = flow_to_color(flow_05xref.cpu().numpy())
        # plt.imsave(filename_string + "17flow_05xref.png", flow_img05xref) 

        # slant_to_color(refined_tile16x[:,1:2,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile16x[:,2:3,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "21slantx_16x.png")
        # slant_to_color(tile_update8x[4].squeeze(0).squeeze(0).cpu().numpy(), tile_update8x[6].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "22slantx_16x.png")
        # slant_to_color(tile_update4x[4].squeeze(0).squeeze(0).cpu().numpy(), tile_update4x[6].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "23slantx_16x.png")
        # slant_to_color(tile_update2x[4].squeeze(0).squeeze(0).cpu().numpy(), tile_update2x[6].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "24slantx_16x.png")
        # slant_to_color(tile_update1x[4].squeeze(0).squeeze(0).cpu().numpy(), tile_update1x[6].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "25slantx_16x.png")
        # slant_to_color(refined_tile1x[:,1:2,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile1x[:,2:3,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "26slantx_16x.png")
        # slant_to_color(refined_tile05x[:,1:2,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile05x[:,2:3,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "27slantx_16x.png")

        # slant_to_color(refined_tile16x[:,17:18,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile16x[:,18:19,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "31slanty_16x.png")
        # slant_to_color(tile_update8x[12].squeeze(0).squeeze(0).cpu().numpy(), tile_update8x[14].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "32slanty_16x.png")
        # slant_to_color(tile_update4x[12].squeeze(0).squeeze(0).cpu().numpy(), tile_update4x[14].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "33slanty_16x.png")
        # slant_to_color(tile_update2x[12].squeeze(0).squeeze(0).cpu().numpy(), tile_update2x[14].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "34slanty_16x.png")
        # slant_to_color(tile_update1x[12].squeeze(0).squeeze(0).cpu().numpy(), tile_update1x[14].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "35slanty_16x.png")
        # slant_to_color(refined_tile1x[:,17:18,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile1x[:,18:19,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "36slanty_16x.png")
        # slant_to_color(refined_tile05x[:,17:18,:,:].squeeze(0).squeeze(0).cpu().numpy(), refined_tile05x[:,18:19,:,:].squeeze(0).squeeze(0).cpu().numpy(), filename_string + "37slanty_16x.png")

        # if self.training:
        fx16 = self.prop_disp_upsample64x(refined_tile16x[:, :1, :, :], refined_tile16x[:, 1:2, :, :], refined_tile16x[:, 2:3, :, :])
        fx8_cur = self.prop_disp_upsample32x(tile_update8x[1], tile_update8x[3], tile_update8x[5])
        fx8_pre = self.prop_disp_upsample32x(tile_update8x[2], tile_update8x[4], tile_update8x[6])
        fx4_cur = self.prop_disp_upsample16x(tile_update4x[1], tile_update4x[3], tile_update4x[5])
        fx4_pre = self.prop_disp_upsample16x(tile_update4x[2], tile_update4x[4], tile_update4x[6])
        fx2_cur = self.prop_disp_upsample8x(tile_update2x[1], tile_update2x[3], tile_update2x[5])
        fx2_pre = self.prop_disp_upsample8x(tile_update2x[2], tile_update2x[4], tile_update2x[6])
        fx1_cur = self.prop_disp_upsample4x(tile_update1x[1], tile_update1x[3], tile_update1x[5])
        fx1_pre = self.prop_disp_upsample4x(tile_update1x[2], tile_update1x[4], tile_update1x[6])
        fx1 = self.prop_disp_upsample4x(refined_tile1x[:, :1, :, :], refined_tile1x[:, 1:2, :, :], refined_tile1x[:, 2:3, :, :])
        fx05 = self.prop_disp_upsample2x(refined_tile05x[:, :1, :, :], refined_tile05x[:, 1:2, :, :], refined_tile05x[:, 2:3, :, :])
        fx_pyramid = [
            fx16,
            fx8_cur,
            fx8_pre,
            fx4_cur,
            fx4_pre,
            fx2_cur,
            fx2_pre,
            fx1_cur,
            fx1_pre,
            fx1,
            fx05,
            final_fx
        ]
        # WARNING: EACH PYRAMID MUST ALIGN ACCORDING TO PRE-CUR ORDER AND RESOLUTION ORDER SINCE SUPERVISION WOULDN'T SEE THE ORDER

        fxx16 = self.dxdy_upsample64x(refined_tile16x[:, 1:2, :, :])
        fxx8_cur = self.dxdy_upsample32x(tile_update8x[3])
        fxx8_pre = self.dxdy_upsample32x(tile_update8x[4])
        fxx4_cur = self.dxdy_upsample16x(tile_update4x[3])
        fxx4_pre = self.dxdy_upsample16x(tile_update4x[4])
        fxx2_cur = self.dxdy_upsample8x(tile_update2x[3])
        fxx2_pre = self.dxdy_upsample8x(tile_update2x[4])
        fxx1_cur = self.dxdy_upsample4x(tile_update1x[3])
        fxx1_pre = self.dxdy_upsample4x(tile_update1x[4])
        fxx1 = self.dxdy_upsample4x(refined_tile1x[:, 1:2, :, :])
        fxx05 = self.dxdy_upsample2x(refined_tile05x[:, 1:2, :, :])
        fxx_pyramid = [
            fxx16,
            fxx8_cur,
            fxx8_pre,
            fxx4_cur,
            fxx4_pre,
            fxx2_cur,
            fxx2_pre,
            fxx1_cur,
            fxx1_pre,
            fxx1,
            fxx05,
        ]

        fxy16 = self.dxdy_upsample64x(refined_tile16x[:, 2:3, :, :])
        fxy8_cur = self.dxdy_upsample32x(tile_update8x[5])
        fxy8_pre = self.dxdy_upsample32x(tile_update8x[6])
        fxy4_cur = self.dxdy_upsample16x(tile_update4x[5])
        fxy4_pre = self.dxdy_upsample16x(tile_update4x[6])
        fxy2_cur = self.dxdy_upsample8x(tile_update2x[5])
        fxy2_pre = self.dxdy_upsample8x(tile_update2x[6])
        fxy1_cur = self.dxdy_upsample4x(tile_update1x[5])
        fxy1_pre = self.dxdy_upsample4x(tile_update1x[6])
        fxy1 = self.dxdy_upsample4x(refined_tile1x[:, 2:3, :, :])
        fxy05 = self.dxdy_upsample2x(refined_tile05x[:, 2:3, :, :])
        fxy_pyramid = [
            fxy16,
            fxy8_cur,
            fxy8_pre,
            fxy4_cur,
            fxy4_pre,
            fxy2_cur,
            fxy2_pre,
            fxy1_cur,
            fxy1_pre,
            fxy1,
            fxy05,
        ]

        conf8_cur = self.dxdy_upsample32x(tile_update8x[7])
        conf8_pre = self.dxdy_upsample32x(tile_update8x[8])
        conf4_cur = self.dxdy_upsample16x(tile_update4x[7])
        conf4_pre = self.dxdy_upsample16x(tile_update4x[8])
        conf2_cur = self.dxdy_upsample8x(tile_update2x[7])
        conf2_pre = self.dxdy_upsample8x(tile_update2x[8])
        conf1_cur = self.dxdy_upsample4x(tile_update1x[7])
        conf1_pre = self.dxdy_upsample4x(tile_update1x[8])
        w_pyramid = [
            conf8_cur,
            conf8_pre,
            conf4_cur,
            conf4_pre,
            conf2_cur,
            conf2_pre,
            conf1_cur,
            conf1_pre,
        ]

        fy16 = self.prop_disp_upsample64x(refined_tile16x[:, 16:17, :, :], refined_tile16x[:, 17:18, :, :], refined_tile16x[:, 18:19, :, :])
        fy8_cur = self.prop_disp_upsample32x(tile_update8x[9], tile_update8x[11], tile_update8x[13])
        fy8_pre = self.prop_disp_upsample32x(tile_update8x[10], tile_update8x[12], tile_update8x[14])
        fy4_cur = self.prop_disp_upsample16x(tile_update4x[9], tile_update4x[11], tile_update4x[13])
        fy4_pre = self.prop_disp_upsample16x(tile_update4x[10], tile_update4x[12], tile_update4x[14])
        fy2_cur = self.prop_disp_upsample8x(tile_update2x[9], tile_update2x[11], tile_update2x[13])
        fy2_pre = self.prop_disp_upsample8x(tile_update2x[10], tile_update2x[12], tile_update2x[14])
        fy1_cur = self.prop_disp_upsample4x(tile_update1x[9], tile_update1x[11], tile_update1x[13])
        fy1_pre = self.prop_disp_upsample4x(tile_update1x[10], tile_update1x[12], tile_update1x[14])
        fy1 = self.prop_disp_upsample4x(refined_tile1x[:, 16:17, :, :], refined_tile1x[:, 17:18, :, :], refined_tile1x[:, 18:19, :, :])
        fy05 = self.prop_disp_upsample2x(refined_tile05x[:, 16:17, :, :], refined_tile05x[:, 17:18, :, :], refined_tile05x[:, 18:19, :, :])
        fy_pyramid = [
            fy16,
            fy8_cur,
            fy8_pre,
            fy4_cur,
            fy4_pre,
            fy2_cur,
            fy2_pre,
            fy1_cur,
            fy1_pre,
            fy1,
            fy05,
            final_fy
        ]

        fyx16 = self.dxdy_upsample64x(refined_tile16x[:, 17:18, :, :])
        fyx8_cur = self.dxdy_upsample32x(tile_update8x[11])
        fyx8_pre = self.dxdy_upsample32x(tile_update8x[12])
        fyx4_cur = self.dxdy_upsample16x(tile_update4x[11])
        fyx4_pre = self.dxdy_upsample16x(tile_update4x[12])
        fyx2_cur = self.dxdy_upsample8x(tile_update2x[11])
        fyx2_pre = self.dxdy_upsample8x(tile_update2x[12])
        fyx1_cur = self.dxdy_upsample4x(tile_update1x[11])
        fyx1_pre = self.dxdy_upsample4x(tile_update1x[12])
        fyx1 = self.dxdy_upsample4x(refined_tile1x[:, 17:18, :, :])
        fyx05 = self.dxdy_upsample2x(refined_tile05x[:, 17:18, :, :])
        fyx_pyramid = [
            fyx16,
            fyx8_cur,
            fyx8_pre,
            fyx4_cur,
            fyx4_pre,
            fyx2_cur,
            fyx2_pre,
            fyx1_cur,
            fyx1_pre,
            fyx1,
            fyx05,
        ]

        fyy16 = self.dxdy_upsample64x(refined_tile16x[:, 18:19, :, :])
        fyy8_cur = self.dxdy_upsample32x(tile_update8x[13])
        fyy8_pre = self.dxdy_upsample32x(tile_update8x[14])
        fyy4_cur = self.dxdy_upsample16x(tile_update4x[13])
        fyy4_pre = self.dxdy_upsample16x(tile_update4x[14])
        fyy2_cur = self.dxdy_upsample8x(tile_update2x[13])
        fyy2_pre = self.dxdy_upsample8x(tile_update2x[14])
        fyy1_cur = self.dxdy_upsample4x(tile_update1x[13])
        fyy1_pre = self.dxdy_upsample4x(tile_update1x[14])
        fyy1 = self.dxdy_upsample4x(refined_tile1x[:, 18:19, :, :])
        fyy05 = self.dxdy_upsample2x(refined_tile05x[:, 18:19, :, :])
        fyy_pyramid = [
            fyy16,
            fyy8_cur,
            fyy8_pre,
            fyy4_cur,
            fyy4_pre,
            fyy2_cur,
            fyy2_pre,
            fyy1_cur,
            fyy1_pre,
            fyy1,
            fyy05,
        ]

        if self.training:
            outputs = {
                "init_cv_pyramid": init_cv_pyramid,
                "fx_pyramid": fx_pyramid,
                "fxx_pyramid": fxx_pyramid,
                "fxy_pyramid": fxy_pyramid,
                "w_pyramid": w_pyramid,
                "fy_pyramid": fy_pyramid,
                "fyx_pyramid": fyx_pyramid,
                "fyy_pyramid": fyy_pyramid
            }

            return outputs

        else:
            return {
                "final_fx": final_fx,
                "final_fy": final_fy,
                "init_cv_pyramid": init_cv_pyramid,
                "fx_pyramid": fx_pyramid,
                "fxx_pyramid": fxx_pyramid,
                "fxy_pyramid": fxy_pyramid,
                "w_pyramid": w_pyramid,
                "fy_pyramid": fy_pyramid,
                "fyx_pyramid": fyx_pyramid,
                "fyy_pyramid": fyy_pyramid,
            }
