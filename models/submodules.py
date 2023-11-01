import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time


class DispUpsampleBySlantedPlane(nn.Module):
    def __init__(self, upscale, ts=4):
        super(DispUpsampleBySlantedPlane, self).__init__()
        self.upscale = upscale
        self.center = (upscale - 1) / 2
        self.DUC = nn.PixelShuffle(upscale)
        self.ts = ts

    def forward(self, tile_disp, tile_dx, tile_dy):
        tile_disp = tile_disp * (self.upscale / self.ts)
        disp0 = []  # for each pixel, upsampled disps are stored in channel dimension
        for i in range(self.upscale):
            for j in range(self.upscale):
                disp0.append(tile_disp + (i - self.center) * tile_dx + (j - self.center) * tile_dy)
        disp0 = torch.cat(disp0, 1)  # [B, upscale**2, H/upscale, W/upscale]
        disp1 = self.DUC(disp0)  # [B, 1, H/1, W/1]
        return disp1


class SlantDUpsampleBySlantedPlaneT4T4(nn.Module):
    """
    Slant map upsampling, input tile size = 4x4, output tile size = 4x4
    """
    def __init__(self, upscale):
        super(SlantDUpsampleBySlantedPlaneT4T4, self).__init__()
        self.upscale = upscale
        self.center = 4 * (upscale - 1) / 2
        self.DUC = nn.PixelShuffle(upscale)

    def forward(self, tile_disp, tile_dx, tile_dy):
        tile_disp = tile_disp * self.upscale
        disp0 = []  # for each pixel, upsampled disps are stored in channel dimension
        for i in range(self.upscale):
            for j in range(self.upscale):
                disp0.append(tile_disp + (i * 4 - self.center) * tile_dx + (j * 4 - self.center) * tile_dy)
        disp0 = torch.cat(disp0, 1)  # [B, upscale**2, H/upscale, W/upscale]
        disp1 = self.DUC(disp0)  # [B, 1, H/1, W/1]
        return disp1


class SlantD2xUpsampleBySlantedPlaneT4T2(nn.Module):
    """
    Slant map upsampling 2x, input tile size = 4x4, output tile size = 2x2
    """
    def __init__(self):
        super(SlantD2xUpsampleBySlantedPlaneT4T2, self).__init__()
        self.DUC = nn.PixelShuffle(2)

    def forward(self, tile_disp, tile_dx, tile_dy):
        disp0 = []  # for each pixel, upsampled disps are stored in channel dimension
        for i in range(2):
            for j in range(2):
                disp0.append(tile_disp + (i * 2 - 1) * tile_dx + (j * 2 - 1) * tile_dy)
        disp0 = torch.cat(disp0, 1)  # [B, upscale**2, H/upscale, W/upscale]
        disp1 = self.DUC(disp0)  # [B, 1, H/1, W/1]
        return disp1


class BuildVolume4d(nn.Module):
    def __init__(self):
        super(BuildVolume4d, self).__init__()

    def forward(self, feat_l, feat_r):
        B, C, H_L, W_L = feat_l.shape
        B, C, H_R, W_R = feat_r.shape

        if torch.cuda.device_count() > 1 and W_L > 200:
            device = torch.device("cuda:1")
            feat_l = feat_l.to(device)
            feat_r = feat_r.to(device)

        feat_l = feat_l.view(B, C, H_L*W_L)
        feat_r = feat_r.view(B, C, H_R*W_R)
        corr = torch.matmul(feat_l.transpose(1,2), feat_r)

        corr = corr.view(B, H_L, W_L, H_R, W_R)
        return corr.contiguous()
        # return corr


class BuildVolumeTilesFlow(nn.Module):
    def __init__(self):
        super(BuildVolumeTilesFlow, self).__init__()

    def forward(self, fea_l, fea_r, flow):
        warped_fea_r = warp(fea_r, flow)
        volume = torch.norm(fea_l - warped_fea_r, 1, 1)
        volume = volume.unsqueeze(1)
        volume = volume.contiguous()
        return volume


class ArgminVolume4d(nn.Module):
    def __init__(self):
        super(ArgminVolume4d, self).__init__()

    def forward(self, volume, scale_x, scale_y):
        # volume: B*H*W*H_CV*W_CV
        # flow: B*H*W*2
        # flow_cost: B*H*W

        B, H, W, H_CV, W_CV = volume.shape

        if torch.cuda.device_count() > 1 and W > 200:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")

        flow_cost, flattened_index = volume.view(B,H,W,-1).max(3)

        flow_x_offset = torch.arange(0,W).to(device).repeat(H, 1)
        flow_y_offset = torch.arange(0,H).to(device).unsqueeze(1).repeat(1,W)

        u = -((flattened_index % W_CV) * scale_x - flow_x_offset).unsqueeze(3)
        v = -((flattened_index // W_CV) * scale_y - flow_y_offset).unsqueeze(3)

        flow = torch.cat((u,v), 3)

        if torch.cuda.device_count() > 1 and W > 200:
            device = torch.device("cuda:0")
            flow = flow.to(device)
            flow_cost = flow_cost.to(device)

        return flow, flow_cost

class DistributedVolumeArgmin(nn.Module):
    def __init__(self):
        super(DistributedVolumeArgmin, self).__init__()

    def forward(self, feat_l, feat_r, scale_x, scale_y):
        B, C, H_L, W_L = feat_l.shape
        B, C, H_R, W_R = feat_r.shape

        if torch.cuda.device_count() > 6 and W_L > 200:
            device1 = torch.device("cuda:0")
            device2 = torch.device("cuda:1")
            device3 = torch.device("cuda:2")
            device4 = torch.device("cuda:3")
            device5 = torch.device("cuda:4")
            device6 = torch.device("cuda:5")
            device7 = torch.device("cuda:6")

            step = H_R//6

            feat_l = feat_l.to(device2).view(B, C, H_L*W_L)
            feat_r1 = feat_r[:,:,0:step,:].to(device2).view(B, C, -1)
            corr1 = torch.matmul(feat_l.transpose(1,2), feat_r1)
            volume1 = corr1.view(B, H_L, W_L, H_R//6, W_R)

            feat_l = feat_l.to(device3).view(B, C, H_L*W_L)
            feat_r2 = feat_r[:,:,step:2*step,:].to(device3).view(B, C, -1)
            corr2 = torch.matmul(feat_l.transpose(1,2), feat_r2)
            volume2 = corr2.view(B, H_L, W_L, H_R//6, W_R)

            feat_l = feat_l.to(device4).view(B, C, H_L*W_L)
            feat_r3 = feat_r[:,:,2*step:3*step,:].to(device4).view(B, C, -1)
            corr3 = torch.matmul(feat_l.transpose(1,2), feat_r3)
            volume3 = corr3.view(B, H_L, W_L, H_R//6, W_R)

            feat_l = feat_l.to(device5).view(B, C, H_L*W_L)
            feat_r4 = feat_r[:,:,3*step:4*step,:].to(device5).view(B, C, -1)
            corr4 = torch.matmul(feat_l.transpose(1,2), feat_r4)
            volume4 = corr4.view(B, H_L, W_L, H_R//6, W_R)

            feat_l = feat_l.to(device6).view(B, C, H_L*W_L)
            feat_r5 = feat_r[:,:,4*step:5*step,:].to(device6).view(B, C, -1)
            corr5 = torch.matmul(feat_l.transpose(1,2), feat_r5)
            volume5 = corr5.view(B, H_L, W_L, H_R//6, W_R)

            feat_l = feat_l.to(device7).view(B, C, H_L*W_L)
            feat_r6 = feat_r[:,:,5*step:6*step,:].to(device7).view(B, C, -1)
            corr6 = torch.matmul(feat_l.transpose(1,2), feat_r6)
            volume6 = corr6.view(B, H_L, W_L, H_R//6, W_R)

            H = H_L
            W = W_L
            W_CV = W_R

            flow_cost1, flattened_index1 = volume1.view(B,H,W,-1).max(3)
            flow_cost2, flattened_index2 = volume2.view(B,H,W,-1).max(3)
            flow_cost3, flattened_index3 = volume3.view(B,H,W,-1).max(3)
            flow_cost4, flattened_index4 = volume4.view(B,H,W,-1).max(3)
            flow_cost5, flattened_index5 = volume5.view(B,H,W,-1).max(3)
            flow_cost6, flattened_index6 = volume6.view(B,H,W,-1).max(3)

            del corr1, corr2, corr3, corr4, corr5, corr6
            del volume1, volume2, volume3, volume4, volume5, volume6
            torch.cuda.empty_cache()

            flow_cost1 = flow_cost1.to(device1)
            flow_cost2 = flow_cost2.to(device1)
            flow_cost3 = flow_cost3.to(device1)
            flow_cost4 = flow_cost4.to(device1)
            flow_cost5 = flow_cost5.to(device1)
            flow_cost6 = flow_cost6.to(device1)

            flattened_index1 = flattened_index1.to(device1)
            flattened_index2 = flattened_index2.to(device1)
            flattened_index3 = flattened_index3.to(device1)
            flattened_index4 = flattened_index4.to(device1)
            flattened_index5 = flattened_index5.to(device1)
            flattened_index6 = flattened_index6.to(device1)

            flattened_index2 = flattened_index2 + (step * W)
            flattened_index3 = flattened_index3 + (2*step * W)
            flattened_index4 = flattened_index4 + (3*step * W)
            flattened_index5 = flattened_index5 + (4*step * W)
            flattened_index6 = flattened_index6 + (5*step * W)

            flow_cost = torch.cat((flow_cost1.unsqueeze(3), flow_cost2.unsqueeze(3), flow_cost3.unsqueeze(3), flow_cost4.unsqueeze(3), flow_cost5.unsqueeze(3), flow_cost6.unsqueeze(3)), 3)
            flattened_index = torch.cat((flattened_index1.unsqueeze(3), flattened_index2.unsqueeze(3), flattened_index3.unsqueeze(3), flattened_index4.unsqueeze(3), flattened_index5.unsqueeze(3), flattened_index6.unsqueeze(3)), 3)
            flow_cost, index = flow_cost.max(3)
            index = index.unsqueeze(3)
            flattened_index = torch.gather(flattened_index, 3, index)
            flattened_index = flattened_index.squeeze(3)
            
            flow_x_offset = torch.arange(0,W).to(device1).repeat(H, 1)
            flow_y_offset = torch.arange(0,H).to(device1).unsqueeze(1).repeat(1,W)

            u = -((flattened_index % W_CV) * scale_x - flow_x_offset).unsqueeze(3)
            v = -((flattened_index // W_CV) * scale_y - flow_y_offset).unsqueeze(3)

            flow = torch.cat((u,v), 3)

        else:
            device = torch.device("cuda:0")
            feat_l = feat_l.view(B, C, H_L*W_L)
            feat_r = feat_r.view(B, C, H_R*W_R)
            corr = torch.matmul(feat_l.transpose(1,2), feat_r)
            volume = corr.view(B, H_L, W_L, H_R, W_R)

            B, H, W, H_CV, W_CV = volume.shape

            flow_cost, flattened_index = volume.view(B,H,W,-1).max(3)
            flow_x_offset = torch.arange(0,W).to(device).repeat(H, 1)
            flow_y_offset = torch.arange(0,H).to(device).unsqueeze(1).repeat(1,W)

            u = -((flattened_index % W_CV) * scale_x - flow_x_offset).unsqueeze(3)
            v = -((flattened_index // W_CV) * scale_y - flow_y_offset).unsqueeze(3)

            flow = torch.cat((u,v), 3)

        return flow, flow_cost


def warp(x, flow):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flow

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    flow = flow.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(x, vgrid)
    # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    # mask = F.grid_sample(mask, vgrid)

    # mask[mask <0.9999] = 0
    # mask[mask >0] = 1

    # return output*mask
    return output
    