import torch
import torch.nn.functional as F
import pdb


def init_loss(pred_init_cost: torch.Tensor, d_gt: torch.Tensor, maxdisp, beta=1):
    """
    Initialization loss, HITNet paper eqt(10
    :param pred_init_cost:
    :param d_gt:
    :param beta:
    :return: init loss [B*1*H*W]
    """
    cost_gt = subpix_cost(pred_init_cost, d_gt, maxdisp)
    cost_nm = torch.gather(pred_init_cost, 1, get_non_match_disp(pred_init_cost, d_gt))
    loss = cost_gt + F.relu(beta - cost_nm)
    # print("test")
    # print(loss.shape)
    # print(loss)
    # pdb.set_trace()
    return loss


def subpix_cost(cost: torch.Tensor, disp: torch.Tensor, maxdisp: int):
    """
    phi, e.g. eqt(9) in HITNet paper
    :param cost:
    :param disp:
    :return:
    """
    # pdb.set_trace()
    disp[disp >= maxdisp - 1] = maxdisp - 2
    disp[disp < 0] = 0
    disp_floor = disp.floor()
    sub_cost = (disp - disp_floor) * torch.gather(cost, 1, disp_floor.long()+1) + (disp_floor + 1 - disp) * torch.gather(cost, 1, disp_floor.long())
    # pdb.set_trace()
    return sub_cost



def get_non_match_disp(pred_init_cost: torch.Tensor, fx_gt: torch.Tensor, fy_gt: torch.Tensor):
    """
    HITNet paper, eqt (11)
    :param pred_init_cost: B, D, H, W
    :param d_gt: B, 1, H, W
    :return: LongTensor: min_non_match_disp: B, 1, H, W
    """

    B, H, W, H_CV, W_CV = pred_init_cost.size()
    scale = H / H_CV

    fx_cand = torch.arange(0, W_CV, step=1, device=fx_gt.device).view(1, 1, 1, -1).repeat(B, H, W, 1).float()
    fy_cand = torch.arange(0, H_CV, step=1, device=fy_gt.device).view(1, 1, 1, -1).repeat(B, H, W, 1).float()

    match_fx_lower_bound = fx_gt - 1.5
    match_fx_upper_bound = fx_gt + 1.5
    match_fy_lower_bound = fy_gt - 1.5
    match_fy_upper_bound = fy_gt + 1.5

    INF_fx = torch.Tensor([float("Inf")]).view(1, 1, 1, 1).repeat(B, H, W, W_CV).to(fx_gt.device)
    INF_fy = torch.Tensor([float("Inf")]).view(1, 1, 1, 1).repeat(B, H, W, H_CV).to(fy_gt.device)
    INF_4D = torch.Tensor([float("Inf")]).view(1, 1, 1, 1, 1).repeat(B, H, W, H_CV, W_CV).to(fx_gt.device)

    tmp_cost_fx = torch.where((fx_cand < match_fx_lower_bound) | (fx_cand > match_fx_upper_bound), fx_cand, INF_fx)
    tmp_cost_fy = torch.where((fy_cand < match_fy_lower_bound) | (fy_cand > match_fy_upper_bound), fy_cand, INF_fy)

    tmp_cost_fx = tmp_cost_fx.unsqueeze(3).repeat(1, 1, 1, H_CV, 1)
    tmp_cost_fy = tmp_cost_fy.unsqueeze(4).repeat(1, 1, 1, 1, W_CV)

    tmp_cost = torch.where((tmp_cost_fx != INF_4D) & (tmp_cost_fy != INF_4D), pred_init_cost, INF_4D)

    _, flattened_index = tmp_cost.view(B,H,W,-1).max(3)
    flow_x_offset = torch.arange(0,W).cuda().repeat(H, 1)
    flow_y_offset = torch.arange(0,H).cuda().unsqueeze(1).repeat(1,W)
    u = -((flattened_index % W_CV) * scale - flow_x_offset).unsqueeze(1)
    v = -((flattened_index // W_CV) * scale - flow_y_offset).unsqueeze(1)

    flow = torch.cat((u,v), 1)
    return flow

#
# if __name__ == '__main__':
#     cost = torch.rand(1, 12, 2, 2)
#     d_gt = torch.rand(1, 1, 2, 2)*4
#     output_cost = init_loss(cost, d_gt)
#     pdb.set_trace()


