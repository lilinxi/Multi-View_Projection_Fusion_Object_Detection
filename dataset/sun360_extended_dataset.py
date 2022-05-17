import os
import logging

import cv2
import numpy as np

import torch.utils.data

import proto_gen.detect_pb2

# Sun360 extended dataset
Sun360ExtendedClassLabels = ['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                             'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"]
Sun360ExtendedPanoWidth = 1024
Sun360ExtendedPanoHeight = 512


class Sun360ExtendedDataset(torch.utils.data.Dataset):
    width_099_count = 0 # 疑似错误标注，标注宽度大于 90% 的图片数量

    def __init__(self, dataset_root: str, train: bool) -> None:
        super().__init__()

        # 1. 初始化数据集根目录和数据变换
        self.dataset_root = os.path.join(dataset_root, "train") if train else os.path.join(dataset_root, "test")
        self.train = train
        self.class_labels = Sun360ExtendedClassLabels

        # 2. 对所有的文件路径进行排序，确保图像和蒙版对齐
        self.images_dir = os.path.join(self.dataset_root, "rgb")
        self.masks_dir = os.path.join(self.dataset_root, "masks")
        self.images_name = list(sorted(os.listdir(self.images_dir)))  # images 列表
        self.masks_name = list(sorted(os.listdir(self.masks_dir)))  # masks 列表
        self.image_masks_name = list()  # images->list[masks] 列表

        mask_index = 0
        for image_index, image_name in enumerate(self.images_name):
            self.image_masks_name.append([])
            image_name = image_name.split(".")[0]
            # 遍历所有 mask
            while mask_index < len(self.masks_name) \
                    and self.masks_name[mask_index].startswith(image_name):  # 该 mask 属于该 image
                self.image_masks_name[image_index].append(self.masks_name[mask_index])
                mask_index += 1

    def __getitem__(self, idx: int) -> proto_gen.detect_pb2.DatasetModel:
        # ground_truth = open(
        #     os.path.join(
        #         "/Users/bytedance/Dataset",
        #         "sun360_extended_dataset_format",
        #         "labels",
        #         "train" if self.train else "test",
        #         self.images_name[idx].split(".")[0] + ".txt"
        #     ),
        #     "w"
        # )
        # 1. 拼接文件路径
        image_path = os.path.join(self.images_dir, self.images_name[idx])
        image_filename = image_path.split(os.sep)[-1].split(".")[0]
        ground_truth_bbx_list = []
        for image_mask_name in self.image_masks_name[idx]:
            image_mask_path = os.path.join(self.masks_dir, image_mask_name)
            mask = cv2.imread(image_mask_path)
            mask_gray = mask[:, :, 0] == 255
            pos_yx = np.where(mask_gray)
            # 获取掩码的包围盒的坐标
            xmin = np.min(pos_yx[1])
            xmax = np.max(pos_yx[1])
            ymin = np.min(pos_yx[0])
            ymax = np.max(pos_yx[0])
            class_label = image_mask_name.split(".")[0].split("_")[4]

            label = self.class_labels.index(class_label)
            x_center = (xmax + xmin) / 2 / Sun360ExtendedPanoWidth
            y_center = (ymax + ymin) / 2 / Sun360ExtendedPanoHeight
            width = (xmax - xmin) / Sun360ExtendedPanoWidth
            height = (ymax - ymin) / Sun360ExtendedPanoHeight

            if width > 0.9:
                Sun360ExtendedDataset.width_099_count += 1
                logging.debug(f'掩码标注跨边界，暂时跳过处理: {Sun360ExtendedDataset.width_099_count}, {image_mask_name}')
                pos_xy = np.where(mask[:, :, 0].T == 255)
                pos_x = np.unique(pos_xy[0])
                pixel_width = len(pos_x)
                xmin_mid = None
                xmax_mid = None
                for i in range(Sun360ExtendedPanoWidth):
                    if not xmin_mid:
                        if i in pos_x:
                            continue
                        else:
                            xmin_mid = i
                    else:
                        if i not in pos_x:
                            continue
                        else:
                            xmax_mid = i
                            break
                if xmax_mid == None or pixel_width / Sun360ExtendedPanoWidth > 0.9:
                    logging.warning(f'标注错误: {image_mask_name}')
                    continue
                if xmin_mid - xmin > pixel_width / 3:  # 足够大掩码的可以拆成两个包围盒
                    xmin = xmin
                    xmax = xmin_mid
                    x_center = (xmax + xmin) / 2 / Sun360ExtendedPanoWidth
                    y_center = (ymax + ymin) / 2 / Sun360ExtendedPanoHeight
                    width = (xmax - xmin) / Sun360ExtendedPanoWidth
                    height = (ymax - ymin) / Sun360ExtendedPanoHeight
                    ground_truth_bbx_list.append(proto_gen.detect_pb2.GroundTruthBBX(
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        label=label,

                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        label_name=class_label,
                    ))
                    # ground_truth.write("%s %s %s %s %s\n" % (label, x_center, y_center, width, height))
                if xmax - xmax_mid > pixel_width / 3:  # 足够大掩码的可以拆成两个包围盒
                    xmin = xmax_mid
                    xmax = xmax
                    x_center = (xmax + xmin) / 2 / Sun360ExtendedPanoWidth
                    y_center = (ymax + ymin) / 2 / Sun360ExtendedPanoHeight
                    width = (xmax - xmin) / Sun360ExtendedPanoWidth
                    height = (ymax - ymin) / Sun360ExtendedPanoHeight
                    ground_truth_bbx_list.append(proto_gen.detect_pb2.GroundTruthBBX(
                        xmin=xmax_mid,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        label=label,

                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        label_name=class_label,
                    ))
                    # ground_truth.write("%s %s %s %s %s\n" % (label, x_center, y_center, width, height))
            else:
                ground_truth_bbx_list.append(proto_gen.detect_pb2.GroundTruthBBX(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    label=label,

                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    label_name=class_label,
                ))
                # Each row is class x_center y_center width height format.
                # ground_truth.write("%s %s %s %s %s\n" % (label, x_center, y_center, width, height))

        # 返回索引图像及其标签结果
        return proto_gen.detect_pb2.DatasetModel(
            image_filename=image_filename,
            image_path=image_path,
            ground_truth_bbx_list=ground_truth_bbx_list,
        )

    def __len__(self) -> int:  # 获取数据集的长度
        return len(self.images_name)


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    import utils.plot

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.ERROR,
    )

    for train in [True, False]:
        sun360_dataset = Sun360ExtendedDataset(
            dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset",
            train=train,
        )
        for i in sun360_dataset:
            cv2.imshow(i.image_filename, utils.plot.PlotDatasetModel(dataset_model=i))
            cv2.waitKey(0)
