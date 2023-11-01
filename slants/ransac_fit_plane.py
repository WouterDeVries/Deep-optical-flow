import sys
sys.path.append('./build')
import array_op
import numpy as np
from utils.read_pfm import pfm_imread
from utils.write_pfm import write_pfm
import random
import os


def ransac_fit_plane(src_fn, dst_fp, flow_direction, rs_sigma=1, rs_reg_win=9, rs_nsample=30, rs_iter=100):
    """
    GPU-RANSAC fitting plane for patched in a image on full resolution kitti dataset

    :param src_fn: source image name
    :param dst_fp: destination image file path
    :param rs_sigma: threshold gating inliers
    :param rs_reg_win: regression window (in HITNet, 9x9)
    :param rs_nsample: number of sampled points in a regression window: in general, 3/8 of points in a window
    :param rs_iter: ransac ieration num
    :return: Full resolution fx, fy in pfm format
    """
    flow, _ = pfm_imread(src_fn)
    flow = np.ascontiguousarray(flow, dtype=np.float32)
    flow = np.expand_dims(flow, 0)
    
    sigma = float(rs_sigma)  # threshold gating inliers
    img_h = flow.shape[1]
    img_w = flow.shape[2]
    reg_win = rs_reg_win
    reg_win_rad = (reg_win - 1) // 2
    nsamples = reg_win ** 2
    r_nsample = rs_nsample
    r_iter = rs_iter
    total_pixels = (img_w - reg_win_rad * 2) * (img_h - reg_win_rad * 2)

    # gather pixels for a regression window
    for i in range(-reg_win_rad, reg_win_rad + 1):
        for j in range(-reg_win_rad, reg_win_rad + 1):
            if i == -reg_win_rad and j == -reg_win_rad:
                data = flow[:, :-reg_win_rad + i, :-reg_win_rad + j]
            elif i == reg_win_rad and j == reg_win_rad:
                data = np.concatenate((data, flow[:, reg_win_rad + i:, reg_win_rad + j:]), axis=0)
            elif i == reg_win_rad:
                data = np.concatenate((data, flow[:, reg_win_rad + i:, reg_win_rad + j:-reg_win_rad + j]), axis=0)
            elif j == reg_win_rad:
                data = np.concatenate((data, flow[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:]), axis=0)
            else:
                data = np.concatenate(
                    (data, flow[:, reg_win_rad + i:-reg_win_rad + i, reg_win_rad + j:-reg_win_rad + j]), axis=0)
    data = np.array(data)
    data = data.transpose(1, 2, 0)

    # Generate coordinate for a regression window
    x_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    y_samples = np.arange(-reg_win_rad, reg_win_rad+1, dtype=np.float32)
    ord_x = np.tile(x_samples.reshape(reg_win, 1), (1, reg_win))
    ord_y = np.tile(y_samples.reshape(1, reg_win), (reg_win, 1))

    # container array for plane coefficients, e.g. fx, fy and f
    coef = np.zeros(shape=[total_pixels, 3]).astype('float32')

    # random index for ransac sampling. For each patch of image, the r_samples is exactly the same
    r_samples = []
    for i in range(r_iter):
        sequence = np.arange(nsamples, dtype=int).tolist()
        r_samples.append(random.sample(sequence, r_nsample))
    r_samples = np.array(r_samples)

    # vectorization of all the arrays for kernel operation
    ord_x_vec = ord_x.flatten()
    ord_y_vec = ord_y.flatten()
    data_vec = data.flatten()
    coef_vec = coef.flatten()
    r_samples_vec = r_samples.flatten()

    # run RANSAC_GPU kernel
    array_op.RANSAC_GPU(ord_x_vec, ord_y_vec, data_vec, coef_vec, total_pixels, nsamples, r_samples_vec, r_iter,
                        r_nsample, sigma)

    # fetch the estimated coefficients
    coef = coef_vec.reshape(total_pixels, 3)
    est_fx = coef[:, 0]
    est_fy = coef[:, 1]
    # est_f = coef[:, 2]

    # reshape and edge padding for estimated results
    est_fx = est_fx.reshape((img_h - reg_win_rad * 2), -1)
    est_fy = est_fy.reshape((img_h - reg_win_rad * 2), -1)
    # est_f = est_f.reshape((img_h - reg_win_rad * 2), -1)
    est_fy_image = np.pad(est_fy, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    est_fx_image = np.pad(est_fx, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')
    # est_f_image = np.pad(est_f, pad_width=((reg_win_rad, reg_win_rad), (reg_win_rad, reg_win_rad)), mode='edge')

    # write the estimated results to pfm
    fx_path = os.path.join(dst_fp, flow_direction + 'x')
    fy_path = os.path.join(dst_fp, flow_direction + 'y')
    os.makedirs(fx_path, exist_ok=True)
    os.makedirs(fy_path, exist_ok=True)
    write_pfm(fx_path+'/'+src_fn.split('/')[-1], est_fx_image)
    write_pfm(fy_path+'/'+src_fn.split('/')[-1], est_fy_image)
    # write_pfm('est_f_0751.pfm', est_f_image)
