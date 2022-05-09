import os
import logging
from typing import List

import cv2

import torch.utils.data

import proto_gen.detect_pb2


class FormatDataset(torch.utils.data.Dataset):
    """
    Dataset for the SUN360 extended dataset format.
    读取标准化格式数据集
    """

    def __init__(self,
                 dataset_root: str, train: bool,
                 image_width: int, image_height: int,
                 class_labels: List[str]
                 ) -> None:
        super().__init__()

        # 1. 初始化数据集根目录和数据变换
        self.dataset_root = dataset_root
        self.train = train
        self.images_dir = os.path.join(self.dataset_root, "images", "train" if train else "test")
        self.labels_dir = os.path.join(self.dataset_root, "labels", "train" if train else "test")
        self.image_width = image_width
        self.image_height = image_height
        self.class_labels = class_labels

        # 2. 对所有的文件路径进行排序，确保图像和蒙版对齐
        self.images_name = [
            name.split(os.sep)[-1].split(".")[0]
            for name in
            list(sorted(os.listdir(self.images_dir)))
        ]  # images 列表

    def __getitem__(self, idx: int) -> proto_gen.detect_pb2.DatasetModel:
        # 1. 拼接文件路径
        image_name = self.images_name[idx]
        image_path = os.path.join(self.images_dir, image_name + ".png")
        label_path = os.path.join(self.labels_dir, image_name + ".txt")

        ground_truth_bbx_list = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                label, x_center, y_center, width, height = line.split(" ")
                label = int(label)
                x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
                label_name = self.class_labels[label]
                xmin = (x_center - width / 2) * self.image_width
                xmax = (x_center + width / 2) * self.image_width
                ymin = (y_center - height / 2) * self.image_height
                ymax = (y_center + height / 2) * self.image_height
                ground_truth_bbx_list.append(proto_gen.detect_pb2.GroundTruthBBX(
                    xmin=round(xmin),
                    ymin=round(ymin),
                    xmax=round(xmax),
                    ymax=round(ymax),
                    label=label,

                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    label_name=label_name,
                ))
                logging.debug(f"{label_path}: {label_name} {label} {x_center} {y_center} {width} {height}")

        # 返回索引图像及其标签结果
        return proto_gen.detect_pb2.DatasetModel(
            image_filename=image_name,
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
    import dataset.sun360_extended_dataset

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO,
    )

    format_dataset = FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=True,
        image_width=dataset.sun360_extended_dataset.Sun360ExtendedPanoWidth,
        image_height=dataset.sun360_extended_dataset.Sun360ExtendedPanoHeight,
        class_labels=dataset.sun360_extended_dataset.Sun360ExtendedClassLabels,
    )
    for i, data in enumerate(format_dataset):
        cv2.imshow(data.image_filename, utils.plot.PlotDatasetModel(dataset_model=data))
        cv2.waitKey(0)
