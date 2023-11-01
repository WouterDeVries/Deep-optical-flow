import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms.functional as photometric
import cv2


class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filenames, training):
        self.datapath = datapath
        self.img1_filenames, self.img2_filenames, self.flow_filenames, self.fxx_filenames, self.fxy_filenames, self.fyx_filenames, self.fyy_filenames = self.load_path(list_filenames)
        self.training = training
        if self.training:
            assert self.flow_filenames is not None

    def load_path(self, list_filenames):
        lines = read_all_lines(list_filenames)
        splits = [line.split() for line in lines]
        img1_images = [x[0] for x in splits]
        img2_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return img1_images, img2_images, None, None, None
        else:
            flow_images = [x[2] for x in splits]
            fxx = [x[3] for x in splits]
            fxy = [x[4] for x in splits]
            fyx = [x[5] for x in splits]
            fyy = [x[6] for x in splits]
            return img1_images, img2_images, flow_images, fxx, fxy, fyx, fyy

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_flow(self, filename):
        flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
        flow = flow[:,:,::-1].astype(np.float32)
        flow, valid = flow[:, :, :2], flow[:, :, 2]
        flow = (flow - 2**15) / 64.0
        return flow

    def load_flow_slant(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.img1_filenames)

    def __getitem__(self, index):
        img1 = self.load_image(os.path.join(self.datapath, self.img1_filenames[index]))
        img2 = self.load_image(os.path.join(self.datapath, self.img2_filenames[index]))

        if self.flow_filenames:  # has flow ground truth
            flow = self.load_flow(os.path.join(self.datapath, self.flow_filenames[index]))
        else:
            flow = None

        if self.fxx_filenames and self.fxy_filenames and self.fyx_filenames and self.fyy_filenames:  # has flow slant ground truth
            fxx_gt = self.load_flow_slant(os.path.join(self.datapath, self.fxx_filenames[index]))
            fxy_gt = self.load_flow_slant(os.path.join(self.datapath, self.fxy_filenames[index]))
            fyx_gt = self.load_flow_slant(os.path.join(self.datapath, self.fyx_filenames[index]))
            fyy_gt = self.load_flow_slant(os.path.join(self.datapath, self.fyy_filenames[index]))

        else:
            fxx_gt = None
            fxy_gt = None
            fyx_gt = None
            fyy_gt = None

        # training
        if self.training:
            w, h = img1.size
            crop_w, crop_h = 1152, 320  # similar to crops of HITNet paper, but multiple of 64

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            img1 = img1.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            img2 = img2.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            flow = flow[y1:y1 + crop_h, x1:x1 + crop_w, :]
            fxx_gt = fxx_gt[y1:y1 + crop_h, x1:x1 + crop_w]
            fxy_gt = fxy_gt[y1:y1 + crop_h, x1:x1 + crop_w]
            fyx_gt = fyx_gt[y1:y1 + crop_h, x1:x1 + crop_w]
            fyy_gt = fyy_gt[y1:y1 + crop_h, x1:x1 + crop_w]

            # photometric augmentation: brightness and contrast perturb
            sym_random_brt = np.random.uniform(0.8, 1.2)
            sym_random_cts = np.random.uniform(0.8, 1.2)
            asym_random_brt = np.random.uniform(0.95, 1.05, size=2)
            asym_random_cts = np.random.uniform(0.95, 1.05, size=2)
            # brightness
            img1 = photometric.adjust_brightness(img1, sym_random_brt)
            img2 = photometric.adjust_brightness(img2, sym_random_brt)
            img1 = photometric.adjust_brightness(img1, asym_random_brt[0])
            img2 = photometric.adjust_brightness(img2, asym_random_brt[1])
            # contrast
            img1 = photometric.adjust_contrast(img1, sym_random_cts)
            img2 = photometric.adjust_contrast(img2, sym_random_cts)
            img1 = photometric.adjust_contrast(img1, asym_random_cts[0])
            img2 = photometric.adjust_contrast(img2, asym_random_cts[1])

            # to tensor, normalize
            processed = get_transform()
            img1 = processed(img1)
            img2 = processed(img2)

            # random patch exchange of image 2
            patch_h = random.randint(50, 180)
            patch_w = random.randint(50, 250)
            patch1_x = random.randint(0, crop_h-patch_h)
            patch1_y = random.randint(0, crop_w-patch_w)
            patch2_x = random.randint(0, crop_h-patch_h)
            patch2_y = random.randint(0, crop_w-patch_w)

            img_patch = img2[:, patch2_x:patch2_x + patch_h, patch2_y:patch2_y + patch_w]
            img2[:, patch1_x:patch1_x + patch_h, patch1_y:patch1_y + patch_w] = img_patch

            return {"img1": img1,
                    "img2": img2,
                    "flow": flow,
                    "fxx_gt": fxx_gt,
                    "fxy_gt": fxy_gt,
                    "fyx_gt": fyx_gt,
                    "fyy_gt": fyy_gt}

        # testing
        else:
            w, h = img1.size

            # img1.save("img/current_model/(" + str(index) + ") " + self.img1_filenames[index][30:39] + ".png")
            # img2.save("img/current_model/(" + str(index) + ") " + self.img2_filenames[index][30:39] + ".png")

            # normalize
            processed = get_transform()
            img1 = processed(img1).numpy()
            img2 = processed(img2).numpy()

            # pad to size 1280x384
            top_pad = 384 - h
            right_pad = 1280 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            img1 = np.lib.pad(img1, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            img2 = np.lib.pad(img2, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            # pad flow gt
            if flow is not None:
                assert len(flow.shape) == 3
                flow_x = np.lib.pad(flow[:,:,0], ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                flow_y = np.lib.pad(flow[:,:,1], ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                flow_x = np.expand_dims(flow_x, 2)
                flow_y = np.expand_dims(flow_y, 2)
                flow = np.concatenate((flow_x, flow_y), 2)

            # pad flow slant
            if fxx_gt is not None and fxy_gt is not None and fyx_gt is not None and fyy_gt is not None:
                assert len(fxx_gt.shape) == 2
                fxx_gt = np.lib.pad(fxx_gt, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                assert len(fxy_gt.shape) == 2
                fxy_gt = np.lib.pad(fxy_gt, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                assert len(fyx_gt.shape) == 2
                fyx_gt = np.lib.pad(fyx_gt, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                assert len(fyy_gt.shape) == 2
                fyy_gt = np.lib.pad(fyy_gt, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if flow is not None and fxx_gt is not None and fxy_gt is not None and fyx_gt is not None and fyy_gt is not None:
                return {"img1": img1,
                        "img2": img2,
                        "flow": flow,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "fxx_gt": fxx_gt,
                        "fxy_gt": fxy_gt,
                        "fyx_gt": fyx_gt,
                        "fyy_gt": fyy_gt,
                        "img1_filename": self.img1_filenames[index],
                        "img2_filename": self.img2_filenames[index]}
            else:
                return {"img1": img1,
                        "img2": img2,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "img1_filename": self.img1_filenames[index],
                        "img2_filename": self.img2_filenames[index]}
