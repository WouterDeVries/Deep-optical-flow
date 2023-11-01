import torch
import torch.nn.functional as F
from loss.initialization_loss import init_loss
from loss.propagation_loss import prop_loss, slant_loss, w_loss


def global_loss(init_cv_cost_pyramid, prop_disp_pyramid, dx_pyramid, dy_pyramid, w_pyramid,
                d_gt, dx_gt, dy_gt, maxdisp, prop_diff_pyramid_fx_fy,
                lambda_init=1, lambda_prop=1, lambda_slant=1, lambda_w=1):
    """

    :param init_cv_cost_pyramid:
    :param prop_disp_pyramid:
    :param slant_pyramid:
    :param w_pyramid:
    :param d_gt:
    :param maxdisp:
    :param loss_init:
    :param loss_prop:
    :param loss_slant:
    :param loss_w:
    :param lambda_init:
    :param lambda_prop:
    :param lambda_slant:
    :param lambda_w:
    :return:
    """

    # d_gt_pyramid = []
    # for i in range(len(init_cv_cost_pyramid)):
    #     scale = 4 * (2 ** i)  # 4,8,16,32,64
    #     d_gt_pyramid.append(torch.nn.MaxPool2d(scale, scale)(d_gt)/(scale/4))
    # d_gt_pyramid.reverse()  # disp ground truth generation. From small to large.

    # init_loss_pyramid = []
    # for i, cv in enumerate(init_cv_cost_pyramid):
    #     mask = (d_gt_pyramid[i] > 0) & (d_gt_pyramid[i] < maxdisp/(2**(len(init_cv_cost_pyramid)-1-i)))
    #     init_loss_pyramid.append(
    #         lambda_init * init_loss(cv, d_gt_pyramid[i], maxdisp/(2**(len(init_cv_cost_pyramid)-1-i)))[mask]
    #     )
    # init_loss_vec = torch.cat(init_loss_pyramid, dim=0)  # 1-dim vector

    prop_loss_pyramid = []  # masked
    prop_diff_pyramid = []  # not masked
    mask = (d_gt > -(maxdisp)) & (d_gt < (maxdisp)) & (d_gt != 0)

    prop_loss_weights = [1/64, 1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1]
    for i, disp in enumerate(prop_disp_pyramid):
        prop_diff_pyramid.append(
            torch.abs(d_gt - disp)
        )
        prop_loss_pyramid.append(
            lambda_prop * prop_loss_weights[i] * prop_loss(prop_diff_pyramid[i], 10000)[mask]
        )
    prop_loss_vec = torch.cat(prop_loss_pyramid, dim=0)

    slant_loss_pyramid = []
    slant_loss_weights = [1/64, 1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2]
    for i in range(len(dx_pyramid)):
        slant_loss_pyramid.append(
            lambda_slant * slant_loss_weights[i] * slant_loss(dx_pyramid[i], dy_pyramid[i], dx_gt, dy_gt, prop_diff_pyramid[i], mask)
        )
    slant_loss_vec = torch.cat(slant_loss_pyramid, dim=0)

    w_loss_pyramid = []
    w_loss_weights = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
    for i, w in enumerate(w_pyramid):
        w_loss_pyramid.append(
            lambda_w * w_loss_weights[i] * w_loss(w, prop_diff_pyramid_fx_fy[i+1], mask)  # index for prop_diff_pyramid plus 1 since there is no confidence at 1st level
        )
    w_loss_vec = torch.cat(w_loss_pyramid, dim=0)

    total_loss_list = [prop_loss_vec, slant_loss_vec, w_loss_vec]

    return total_loss_list
