import os
import cv2
import numpy as np
import time
from scipy.interpolate import NearestNDInterpolator
from utils.read_pfm import pfm_imread
from utils.write_pfm import write_pfm


def sparse_flow_interp(s_map):
    """
    Nearest interpolate the missing points in a sparse represented flow map
    :param s_map: sparse map
    :return: dense map
    """
    img_h = s_map.shape[0]
    img_w = s_map.shape[1]
    valid_mask = ((s_map != np.inf) & (s_map!= 0))
    valid_indices = np.argwhere(valid_mask)
    X = np.linspace(0, img_w-1, img_w)
    Y = np.linspace(0, img_h-1, img_h)
    all_X, all_Y = np.meshgrid(X, Y)

    valid_values = s_map[valid_mask]
    interp = NearestNDInterpolator(list(zip(valid_indices[:, 0], valid_indices[:, 1])), valid_values)
    d_map = interp(all_Y, all_X)
    return d_map


def sparse_to_dense(src_data_path, dst_data_path, filename, flow_direction):
    with open(filename) as f:
        flow_gt_fn_lines = [line.rstrip() for line in f.readlines()]

    for i, flow_gt in enumerate(flow_gt_fn_lines):
        st_time = time.time()
        src_fn = os.path.join(src_data_path, flow_gt)

        flow = cv2.imread(src_fn, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
        flow = flow[:,:,::-1].astype(np.float32)
        flow, valid = flow[:, :, :2], flow[:, :, 2]
        flow = (flow - 2**15) / 64.0

        flow_1D = flow[:,:,flow_direction]
        dense_flow = sparse_flow_interp(flow_1D)

        os.makedirs(dst_data_path, exist_ok=True)
        write_pfm(dst_data_path + '/' + src_fn.split('/')[-1].replace('.png', '.pfm'), dense_flow)
        print('{}th finish: '.format(i) + src_fn + '; Time: {:.2f}s'.format(time.time() - st_time))


if __name__ == '__main__':
    # KITTI 2012 fx
    src_data_path = '../datasets/KITTI/kitti_2012'
    dst_data_path = 'KITTI_dense/kitti_2012/fx/'
    filename = 'filenames/kitti12_train.txt'
    flow_direction = 0
    sparse_to_dense(src_data_path, dst_data_path, filename, flow_direction)

    # KITTI 2012 fy
    dst_data_path = 'KITTI_dense/kitti_2012/fy/'
    flow_direction = 1
    sparse_to_dense(src_data_path, dst_data_path, filename, flow_direction)

    # KITTI 2015 fx
    src_data_path = '../datasets/KITTI/kitti_2015'
    dst_data_path = 'KITTI_dense/kitti_2015/fx/'
    filename = 'filenames/kitti15_train.txt'
    flow_direction = 0
    sparse_to_dense(src_data_path, dst_data_path, filename, flow_direction)

    # KITTI 2015 fy
    dst_data_path = 'KITTI_dense/kitti_2015/fy/'
    flow_direction = 1
    sparse_to_dense(src_data_path, dst_data_path, filename, flow_direction)
