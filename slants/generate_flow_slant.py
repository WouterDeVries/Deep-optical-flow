from ransac_fit_plane import ransac_fit_plane
import os
import time


def generate_slant_gt(src_data_path, dst_data_path, filename, flow_direction):
    with open(filename) as f:
        disp_gt_fn_lines = [line.rstrip() for line in f.readlines()]

    for i, disp_gt in enumerate(disp_gt_fn_lines):
        src_fn = os.path.join(src_data_path, disp_gt.replace('.png', '.pfm'))
        dst_fp = os.path.join(dst_data_path, '/'.join(disp_gt.split('/')[1:-1]))
        st_time = time.time()
        ransac_fit_plane(src_fn, dst_fp, flow_direction)
        print('{}th finish: '.format(i)+src_fn+'; Time: {:.2f}s'.format(time.time() - st_time))


if __name__ == '__main__':
    # KITTI 2012 fxx fxy
    src_data_path = 'KITTI_dense/kitti_2012/fx'
    dst_data_path = '../datasets/KITTI/kt12_flow_slant/'
    filename = 'filenames/kitti12_train_dense.txt'
    flow_direction = 'fx'
    generate_slant_gt(src_data_path, dst_data_path, filename, flow_direction)

    # KITTI 2012 fyx fyy
    src_data_path = 'KITTI_dense/kitti_2012/fy'
    flow_direction = 'fy'
    generate_slant_gt(src_data_path, dst_data_path, filename, flow_direction)

    # KITTI 2015 fxx fxy
    src_data_path = 'KITTI_dense/kitti_2015/fx'
    dst_data_path = '../datasets/KITTI/kt15_flow_slant/'
    filename = 'filenames/kitti15_train_dense.txt'
    flow_direction = 'fx'
    generate_slant_gt(src_data_path, dst_data_path, filename, flow_direction)

    # KITTI 2015 fyx fyy
    src_data_path = 'KITTI_dense/kitti_2015/fy'
    flow_direction = 'fy'
    generate_slant_gt(src_data_path, dst_data_path, filename, flow_direction)
