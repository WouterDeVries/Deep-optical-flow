import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .submodules import DispUpsampleBySlantedPlane, BuildVolumeTilesFlow
import time


class TileWarping(nn.Module):
    def __init__(self, args):
        super(TileWarping, self).__init__()
        self.f_up = DispUpsampleBySlantedPlane(4)
        # self.f_up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.build_flow_volume = BuildVolumeTilesFlow()

    def forward(self, tile_plane: torch.Tensor, fea_l: torch.Tensor, fea_r: torch.Tensor):
        """
        local cost volume
        :param tile_plane: d, dx, dy
        :param fea_l:
        :param fea_r:
        :return: local cost volume
        """
        fx = tile_plane[:, 0, :, :].unsqueeze(1)
        fxx = tile_plane[:, 1, :, :].unsqueeze(1)   # h direction
        fxy = tile_plane[:, 2, :, :].unsqueeze(1)  # w direction
        fy = tile_plane[:, 16, :, :].unsqueeze(1)
        fyx = tile_plane[:, 17, :, :].unsqueeze(1)
        fyy = tile_plane[:, 18, :, :].unsqueeze(1)
        local_cv = []

        for flow_x in range(-1,2):
            for flow_y in range(-1,2):
                fx_up = self.f_up(fx + flow_x, fxx, fxy)
                fy_up = self.f_up(fy + flow_y, fyx, fyy)
                # fx_up = self.f_up(fx + flow_x)
                # fy_up = self.f_up(fy + flow_y)
                flow = torch.cat((fx_up, fy_up), 1)
                cost_volume = self.build_flow_volume(fea_l, fea_r, flow)

                local_cv_ws_disp_d = []  # local cost volume in one disp hypothesis [B, 16, H/4, W/4]
                for i in range(4):
                    for j in range(4):
                        local_cv_ws_disp_d.append(cost_volume[:, :, i::4, j::4])
                local_cv_ws_disp_d = torch.cat(local_cv_ws_disp_d, 1)
                local_cv.append(local_cv_ws_disp_d)  # local cost volume containing all the disp hypothesis[B, 144, H/4, W/4]
                # pdb.set_trace()
        local_cv = torch.cat(local_cv, 1)
        return local_cv
