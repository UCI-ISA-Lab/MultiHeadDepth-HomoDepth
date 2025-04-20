import torch
import numpy as np
from skimage.transform import resize, rescale
from matplotlib import pyplot as plt
import time
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider
)
import imageio.v3 as iio
from skimage.transform import resize, rescale
from torch.utils.data import Dataset, DataLoader

import model
import utils
import os



class ADTFromOrg(Dataset):
    def __init__(self, root_dir, train=True, sample_rate=25, initial_list=True, sep_out=False):
        self.root_dir = root_dir
        self.train = train
        self.sample_rate = sample_rate
        self.sep_out = sep_out
        self.sep_list = []

        self.right_id = StreamId("1201-2")
        self.rgb_id = StreamId("214-1")
        if initial_list:
            file_list = []
            if train:
                txt = os.path.join(root_dir, 'train_list.txt')
            else:
                txt = os.path.join(root_dir, 'test_list.txt')
            with open(txt, 'r') as file:
                for line in file:
                    file_list.append(line.strip())
            self.image_list = self.get_file_list(0.7, file_list)

        self.cur_path_provider = None
        self.gt_provider = None
        self.cur_dev = None
        self.cur_vrs = None


    def get_file_list(self, sp, file_list):
        image_list = []
        for level1 in file_list:
            cur_path = os.path.join(self.root_dir, level1)
            paths_provider = AriaDigitalTwinDataPathsProvider(cur_path)
            all_device_serials = paths_provider.get_device_serial_numbers()
            for i, _ in enumerate(all_device_serials):
                paths_provider = AriaDigitalTwinDataPathsProvider(cur_path)
                data_paths = paths_provider.get_datapaths_by_device_num(i)
                gt_provider = AriaDigitalTwinDataProvider(data_paths)
                img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(self.rgb_id)
                img_len = len(img_timestamps_ns)
                for j in range(5, img_len, self.sample_rate):
                    frame = img_timestamps_ns[j]
                    image = gt_provider.get_depth_image_by_timestamp_ns(frame, self.rgb_id)
                    if image.is_valid() and abs(image.dt_ns()) < 1e6:
                        image_list.append([data_paths, frame])
                        if self.sep_out:
                            self.sep_list.append([cur_path, str(i), str(frame)])


        split_index = int(len(image_list) * sp)
        if self.train:
            image_list = image_list[:split_index]
        else:
            image_list = image_list[split_index:]

        print(f"{len(image_list)} data collected")
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        vrs, frame = self.image_list[idx]
        if self.cur_vrs != vrs:
            self.cur_vrs = vrs
            self.gt_provider = AriaDigitalTwinDataProvider(vrs)
            #print('update')

        left_focal, _ = self.gt_provider.get_aria_camera_calibration(self.rgb_id).get_focal_lengths()
        right_focal, _ = self.gt_provider.get_aria_camera_calibration(self.right_id).get_focal_lengths()
        focal_ratio = right_focal / left_focal

        image = self.gt_provider.get_aria_image_by_timestamp_ns(frame, self.rgb_id)
        assert image.is_valid(), "Left image not valid!"
        image = image.data().to_numpy_array()

        sensor_name = self.gt_provider.raw_data_provider_ptr().get_label_from_stream_id(self.rgb_id)
        device_calib = self.gt_provider.raw_data_provider_ptr().get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)
        size, _ = self.gt_provider.get_aria_camera_calibration(self.rgb_id).get_image_size()
        size = int(size * focal_ratio)
        dst_calib = calibration.get_linear_camera_calibration(size, size, right_focal, sensor_name)
        image = calibration.distort_by_calibration(image, dst_calib, src_calib)
        image = image[36:420, 135:423, :]
        left = np.rot90(image, 3)


        image = self.gt_provider.get_depth_image_by_timestamp_ns(frame, self.rgb_id)
        image = image.data().to_numpy_array().astype(np.float32)
        image = calibration.distort_by_calibration(image, dst_calib, src_calib)
        image = image[36:420, 135:423]
        image = np.rot90(image, 3)
        depth = torch.tensor(image.copy()).unsqueeze(0)


        image = self.gt_provider.get_aria_image_by_timestamp_ns(frame, self.right_id)
        assert image.is_valid(), "Right image not valid!"
        image = image.data().to_numpy_array()

        sensor_name = self.gt_provider.raw_data_provider_ptr().get_label_from_stream_id(self.right_id)
        device_calib = self.gt_provider.raw_data_provider_ptr().get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)
        size_x, size_y = self.gt_provider.get_aria_camera_calibration(self.right_id).get_image_size()
        dst_calib = calibration.get_linear_camera_calibration(size_x, size_y, right_focal, sensor_name)
        image = calibration.distort_by_calibration(image, dst_calib, src_calib)
        image = image[-384:, 176:-176]
        right = np.rot90(image, 3)

        if self.sep_out:
            return left, right, depth
        else:
            left_img = torch.tensor(left.copy()).permute(2, 0, 1).float()
            right_img = torch.tensor(right.copy()).repeat(3, 1, 1).float()
            image = torch.cat((left_img, right_img), dim=0)
            return image, depth


    def get_one(self, paths, dev_num, time_seq, rect=True):
        paths = os.path.join(self.root_dir, paths)
        paths_provider = AriaDigitalTwinDataPathsProvider(paths)
        data_paths = paths_provider.get_datapaths_by_device_num(dev_num)
        gt_provider = AriaDigitalTwinDataProvider(data_paths)
        img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(self.rgb_id)
        assert len(img_timestamps_ns) > time_seq, "time_seq exceeds the number of time"
        frame = img_timestamps_ns[time_seq]
        if rect:
            left_focal, _ = gt_provider.get_aria_camera_calibration(self.rgb_id).get_focal_lengths()
            right_focal, _ = gt_provider.get_aria_camera_calibration(self.right_id).get_focal_lengths()
            focal_ratio = right_focal/left_focal

        image = gt_provider.get_aria_image_by_timestamp_ns(frame, self.rgb_id)
        assert image.is_valid(), "Image not valid!"
        image = image.data().to_numpy_array()

        if rect:
            sensor_name = gt_provider.raw_data_provider_ptr().get_label_from_stream_id(self.rgb_id)
            device_calib = gt_provider.raw_data_provider_ptr().get_device_calibration()
            src_calib = device_calib.get_camera_calib(sensor_name)
            size, _ = gt_provider.get_aria_camera_calibration(self.rgb_id).get_image_size()
            size = int(size * focal_ratio)
            dst_calib = calibration.get_linear_camera_calibration(size, size, right_focal, sensor_name)
            image = calibration.distort_by_calibration(image, dst_calib, src_calib)
            image = image[36:420, 135:423, :]
        left_img = np.rot90(image, 3)

        image = gt_provider.get_depth_image_by_timestamp_ns(frame, self.rgb_id)
        assert image.is_valid(), "Image not valid!"
        assert abs(image.dt_ns()) < 1e6, "Time not aligned"
        image = image.data().to_numpy_array().astype(np.float32)
        if rect:
            image = calibration.distort_by_calibration(image, dst_calib, src_calib)
            image = image[36:420, 135:423]
        depth = np.rot90(image, 3)

        image = gt_provider.get_aria_image_by_timestamp_ns(frame, self.right_id)
        assert image.is_valid(), "Image not valid!"
        image = image.data().to_numpy_array()

        if rect:
            sensor_name = gt_provider.raw_data_provider_ptr().get_label_from_stream_id(self.right_id)
            device_calib = gt_provider.raw_data_provider_ptr().get_device_calibration()
            src_calib = device_calib.get_camera_calib(sensor_name)
            size_x, size_y = gt_provider.get_aria_camera_calibration(self.right_id).get_image_size()
            dst_calib = calibration.get_linear_camera_calibration(size_x, size_y, right_focal, sensor_name)
            image = calibration.distort_by_calibration(image, dst_calib, src_calib)
            image = image[-384:, 176:-176]
        right_img = np.rot90(image, 3)

        return left_img, right_img, depth

    def go_through(self, paths, dev_num):
        paths = os.path.join(self.root_dir, paths)
        paths_provider = AriaDigitalTwinDataPathsProvider(paths)
        data_paths = paths_provider.get_datapaths_by_device_num(dev_num)
        gt_provider = AriaDigitalTwinDataProvider(data_paths)
        img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(self.rgb_id)
        for frame in img_timestamps_ns:
            image = gt_provider.get_depth_image_by_timestamp_ns(frame, self.rgb_id)
            assert image.is_valid(), "Image not valid!"
            print(image.dt_ns())

    def extract_images(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if self.train:
            file = 'train'
        else:
            file = 'test'

        if not os.path.exists(os.path.join(dir, file)):
            os.makedirs(os.path.join(dir, file))

        for idx in range(len(self.image_list)):
            vrs, i, frame = self.sep_list[idx]
            vrs = os.path.basename(vrs)
            if not os.path.exists(os.path.join(dir, file, vrs)):
                os.makedirs(os.path.join(dir, file, vrs))
            if not os.path.exists(os.path.join(dir, file, vrs, i)):
                os.makedirs(os.path.join(dir, file, vrs, i))

            cur_file = os.path.join(dir, file, vrs, i, frame)
            if not os.path.exists(cur_file):
                os.makedirs(cur_file)

            left, right, depth = self.__getitem__(idx)
            iio.imwrite(os.path.join(cur_file, 'left.png'), left)
            iio.imwrite(os.path.join(cur_file, 'right.png'), right)
            torch.save(depth, os.path.join(cur_file, 'depth.pt'))


class ADT(Dataset):
    def __init__(self, root_dir, train=True, sample_rate=25, initial_list=True):
        self.root_dir = root_dir
        self.train = train
        self.sample_rate = sample_rate

        self.right_id = StreamId("1201-2")
        self.rgb_id = StreamId("214-1")
        if initial_list:
            file_list = []
            if train:
                txt = os.path.join(root_dir, 'train_list.txt')
            else:
                txt = os.path.join(root_dir, 'test_list.txt')
            with open(txt, 'r') as file:
                for line in file:
                    file_list.append(line.strip())
            self.image_list = self.get_file_list()

        self.cur_path_provider = None
        self.gt_provider = None
        self.cur_dev = None
        self.cur_vrs = None

    def get_file_list(self):
        image_list = []
        if self.train:
            f = 'train'
        else:
            f = 'test'

        for level1 in os.listdir(os.path.join(self.root_dir, f)):
            path1 = os.path.join(self.root_dir, f, level1)
            for level2 in os.listdir(path1):
                path2 = os.path.join(path1, level2)
                for level3 in os.listdir(path2):
                    image_list.append(os.path.join(path2, level3))

        return image_list


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx, sep_out=False):
        path = self.image_list[idx]
        left_dir = os.path.join(path, 'left.png')
        right_dir = os.path.join(path, 'right.png')

        left_img = utils.img2tensor(left_dir, size=(256, 512))
        image = iio.imread(right_dir)
        image = resize(image, (256, 512), anti_aliasing=False, order=0)
        right_img = torch.tensor(image, dtype=left_img.dtype).repeat(3, 1, 1)

        depth = torch.load(os.path.join(path, 'depth.pt'))/60
        #print(depth.shape)
        depth = resize(depth.squeeze().numpy(), (256, 512), anti_aliasing=False, order=0)

        if sep_out:
            return left_img, right_img, depth
        else:
            image = torch.cat((left_img, right_img), dim=0)
            return image, depth



if __name__ == "__main__":
    # INPUT_PATH = '/mnt/nvme_storage_0/dataset/ADT/'
    # train_dataset = ADTFromOrg(INPUT_PATH, train=True, sep_out=True)
    #
    # train_dataset.extract_images('../data/ADT_depth/')
    # val_dataset = ADTFromOrg(INPUT_PATH, train=False, sep_out=True)
    # val_dataset.extract_images('../data/ADT_depth/')

    # paths = "Apartment_release_golden_skeleton_seq100_10s_sample"
    adt = ADT("../data/ADT_depth", train=True)
    # train_loader = DataLoader(adt, batch_size=2, shuffle=False)
    # for i, (images,  disp) in enumerate(train_loader):
    #     print('get 22222')
    left_img, right_img, depth = adt.__getitem__(2)
    plt.subplot(221)
    plt.imshow(left_img)
    plt.subplot(222)
    plt.imshow(right_img)
    plt.subplot(223)
    plt.imshow(depth, cmap='jet')
    plt.show()













