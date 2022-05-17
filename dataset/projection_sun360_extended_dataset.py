import os, sys

sys.path.append(os.getcwd())
import logging
from typing import List

import cv2
import numpy as np

import torch.utils.data

import proto_gen.detect_pb2
import dataset.sun360_extended_dataset


class ProjSun360ExtendedDataset(torch.utils.data.Dataset):
    """
    将 Sun360E 数据集投影，并写入文件
    """

    def __init__(self,
                 dataset_root: str, train: bool,
                 proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams],
                 proj_width: int, proj_height: int,
                 proj_dataset_root: str,
                 proj_func: callable,
                 ) -> None:
        super().__init__()

        # 1. 初始化数据集根目录和数据变换
        self.dataset_root = os.path.join(dataset_root, "train") if train else os.path.join(dataset_root, "test")
        self.train = train
        self.class_labels = dataset.sun360_extended_dataset.Sun360ExtendedClassLabels
        self.proj_params_list = proj_params_list
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.proj_dataset_root = proj_dataset_root
        self.proj_func = proj_func

        # 2. 对所有的文件路径进行排序，确保图像和蒙版对齐
        self.images_dir = os.path.join(self.dataset_root, "rgb")
        self.masks_dir = os.path.join(self.dataset_root, "masks")
        self.images_name = list(sorted(os.listdir(self.images_dir)))  # images 列表
        self.masks_name = list(sorted(os.listdir(self.masks_dir)))  # masks 列表
        self.image_masks_name = list()  # images->list[masks] 列表

        # 3. 将蒙版和图像对应起来
        for image_index, image_name in enumerate(self.images_name):
            self.image_masks_name.append([])
            image_name = image_name.split(".")[0]
            # 遍历所有 mask
            for mask_name in self.masks_name:
                if mask_name.startswith(image_name):  # 该 mask 属于该 image
                    self.image_masks_name[image_index].append(mask_name)

    def __getitem__(self, idx: int) -> np.ndarray:
        # 1. 拼接文件路径
        image_path = os.path.join(self.images_dir, self.images_name[idx])
        image_mask_list = self.image_masks_name[idx]
        image_filename = image_path.split(os.sep)[-1].split(".")[0]
        im = cv2.imread(image_path)
        # 2. 遍历投影参数
        for i, proj_params in enumerate(self.proj_params_list):
            # 3. 投影图像
            proj_image = self.proj_func(
                im, proj_params, proj_width=self.proj_width, proj_height=self.proj_height)
            # 4. 遍历蒙版
            proj_label_list = []
            for ii, image_mask_name in enumerate(image_mask_list):
                # 蒙版对应的标签
                label_name = image_mask_name.split(".")[0].split("_")[4]
                label = self.class_labels.index(label_name)
                # 读取蒙版图像
                image_mask_path = os.path.join(self.masks_dir, image_mask_name)
                image_mask = cv2.imread(image_mask_path)
                # 读取蒙版点阵
                origin_pos = np.where(image_mask[:, :, 0] == 255)
                # 5. 投影蒙版
                proj_image_mask = self.proj_func(
                    image_mask, proj_params, proj_width=self.proj_width, proj_height=self.proj_height)
                # 6. 投影蒙版点阵
                proj_pos = np.where(proj_image_mask[:, :, 0] == 255)
                # 7. 蒙版存在
                if proj_pos[0].shape[0] > 0:
                    # 8. 蒙版比例足够
                    if proj_pos[0].shape[0] / origin_pos[0].shape[0] > 0.3:
                        # cv2.imshow(f"proj_image_mask_{ii}_{label_name}", proj_image_mask)
                        # 9. 获取掩码的包围盒的坐标
                        xmin = np.min(proj_pos[1])
                        xmax = np.max(proj_pos[1])
                        ymin = np.min(proj_pos[0])
                        ymax = np.max(proj_pos[0])

                        x_center = (xmax + xmin) / 2 / self.proj_width
                        y_center = (ymax + ymin) / 2 / self.proj_height
                        width = (xmax - xmin) / self.proj_width
                        height = (ymax - ymin) / self.proj_height

                        # if width > 0.9: # TODO，投影图应该不会跨边界
                        #     logging.warning(f'掩码标注跨边界，暂时跳过处理: {proj_params}, {image_mask_name}')
                        #     continue

                        proj_label_list.append((label, x_center, y_center, width, height))
            # 10. 标签数量达标
            if len(proj_label_list) > 0:
                ground_truth = open(
                    os.path.join(
                        self.proj_dataset_root,
                        "labels",
                        "train" if self.train else "test",
                        f"{image_filename}_{i}.txt"),
                    "w")
                for proj_label in proj_label_list:
                    ground_truth.write(
                        "%s %s %s %s %s\n" %
                        (proj_label[0], proj_label[1], proj_label[2], proj_label[3], proj_label[4])
                    )
                ground_truth.close()

                proj_image_path = os.path.join(
                    self.proj_dataset_root,
                    "images",
                    "train" if self.train else "test",
                    f"{image_filename}_{i}.png")
                cv2.imwrite(proj_image_path, proj_image)

        return im

    def __len__(self) -> int:  # 获取数据集的长度
        return len(self.images_name)


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    import time
    import proj.stereo_proj
    import proj.perspective_proj

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.ERROR,
    )

    proj_params_list = []
    total_theta_step = 2
    for project_dis in range(1, 2):
        for project_size in range(1, 2):
            for theta_step in range(total_theta_step):
                proj_params_list.extend(
                    [
                        proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
                                                                 theta_rotate=theta_step * np.pi * 2 / total_theta_step),
                    ]
                )

    # 19*666*48*0.17/60/60/24=1.4，大概需要一天半
    for train in [True, False]:
        sun360_dataset = ProjSun360ExtendedDataset(
            dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset",
            train=train,
            proj_params_list=proj_params_list,
            proj_width=10,
            proj_height=10,
            proj_dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_proj_tmp",
            # proj_func=proj.stereo_proj.stereo_proj,
            proj_func=proj.perspective_proj.perspective_proj,
        )

        curTime = time.time()
        for i, im in enumerate(sun360_dataset):
            logging.info(f'processing {i} time cost {time.time() - curTime} s')
            curTime = time.time()
            # cv2.imshow('im', im)
            # cv2.waitKey(0)
