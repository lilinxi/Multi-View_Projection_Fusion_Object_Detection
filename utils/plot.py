from typing import List

import cv2
import numpy as np

import proto_gen.detect_pb2
import dataset.sun360_extended_dataset
import utils.nda2bytes


# 调色盘，获得颜色
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


# 绘制数据库输出模型：真值框
def PlotDatasetModel(dataset_model: proto_gen.detect_pb2.DatasetModel, plot_bbx=True) -> np.ndarray:
    if dataset_model.image_ndarray:  # 直接绘制图像
        im = utils.nda2bytes.bytes2ndarray(dataset_model.image_ndarray)
    elif dataset_model.image_path:  # 读取后绘制图像
        im = cv2.imread(dataset_model.image_path)  # BGR
    else:
        raise NotImplementedError
    if plot_bbx:
        tl = round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
        for bbx in dataset_model.ground_truth_bbx_list:
            color = Colors()(bbx.label, True)
            # rect
            c1, c2 = (bbx.xmin, bbx.ymin), (bbx.xmax, bbx.ymax)
            cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            # label
            label = bbx.label_name
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im


# 绘制网络输出模型：检测框
def PlotYolov5ModelResponse(
        yolo_model_resp: proto_gen.detect_pb2.YoloModelResponse,
        class_labels: List[str] = dataset.sun360_extended_dataset.Sun360ExtendedClassLabels
) -> np.ndarray:
    # x, im, color = (128, 128, 128), label = None, line_thickness = 3
    im = cv2.imread(yolo_model_resp.image_path)  # BGR
    tl = round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # tl = round(0.0005 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    for detect_result_bbx in yolo_model_resp.detect_result_bbx_list:
        color = Colors()(detect_result_bbx.label, True)
        # rect
        c1, c2 = (detect_result_bbx.xmin, detect_result_bbx.ymin), (detect_result_bbx.xmax, detect_result_bbx.ymax)
        cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # label
        label = f'{class_labels[detect_result_bbx.label]} {detect_result_bbx.conf:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im


# 同时绘制真值框和检测框
def PlotDatasetModelAndYolov5ModelResponse(
        dataset_model: proto_gen.detect_pb2.DatasetModel,
        detect_resp: proto_gen.detect_pb2.YoloModelResponse,
        class_labels: List[str] = dataset.sun360_extended_dataset.Sun360ExtendedClassLabels) -> np.ndarray:
    if dataset_model.image_ndarray:  # 直接绘制图像
        im = utils.nda2bytes.bytes2ndarray(dataset_model.image_ndarray)
    elif dataset_model.image_path:  # 读取后绘制图像
        im = cv2.imread(dataset_model.image_path)  # BGR
    else:
        raise NotImplementedError
    tl = round(0.001 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    for bbx in dataset_model.ground_truth_bbx_list:
        color = (0, 255, 0)
        # rect
        c1, c2 = (bbx.xmin, bbx.ymin), (bbx.xmax, bbx.ymax)
        cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # label
        label = bbx.label_name
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    for bbx in detect_resp.detect_result_bbx_list:
        color = (0, 0, 255)
        # rect
        c1, c2 = (bbx.xmin, bbx.ymin), (bbx.xmax, bbx.ymax)
        cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # label
        label = class_labels[bbx.label]
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im


# 多张图像网格排列绘制
def plot_nda_list(nda_list: List[np.ndarray], width_size: int = 0, border_size: int = 5) -> np.ndarray:
    """
    Plot a list of ndarrays.
    Args:
        nda_list:  List of ndarrays.
        width_size: Width of the plot. If 0, the width is calculated automatically.

    Returns:
    """
    # 增加边框
    for i in range(len(nda_list)):
        nda_list[i] = cv2.copyMakeBorder(
            nda_list[i],
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT)
    # 正方形排版显示
    if width_size == 0:
        width_size = int(np.ceil(np.sqrt(len(nda_list))))
    height_size = int(np.ceil(len(nda_list) / width_size))
    nda_list_vstack = []
    nda_list_hstack = []
    for i in range(width_size * height_size):
        if i % width_size == 0:  # 换行
            if i > 0:
                nda_list_vstack.append(np.hstack(nda_list_hstack))
                nda_list_hstack = []
        if i >= len(nda_list):
            if i % width_size == 0:  # 正方形正好缺失整行，不补了
                break
            nda_list_hstack.append(np.zeros(nda_list[0].shape, dtype=nda_list[0].dtype))  # 不足的地方补黑
        else:
            nda_list_hstack.append(nda_list[i])  # 添加图片
    # 添加最后一行
    if len(nda_list_hstack) > 0:
        nda_list_vstack.append(np.hstack(nda_list_hstack))
    return np.vstack(nda_list_vstack)
