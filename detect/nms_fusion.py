from typing import List

import numpy as np
import torch

import proto_gen.detect_pb2

from metrics_utils.yolov5_metrics_utils import box_iou


def iou(box1, box2) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    :param box1: x_min, y_min, x_max, y_max
    :param box2: x_min, y_min, x_max, y_max
    :return: iou score
    """
    box1 = torch.unsqueeze(torch.from_numpy(np.array(box1)), 0)
    box2 = torch.unsqueeze(torch.from_numpy(np.array(box2)), 0)

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    iou_matrix = inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
    return iou_matrix.numpy()[0][0]


def nms_fusion_single_label(detect_result_bbx_list: List[proto_gen.detect_pb2.DetectResultBBX]) \
        -> List[proto_gen.detect_pb2.DetectResultBBX]:
    if len(detect_result_bbx_list) == 0:
        return []
    label = detect_result_bbx_list[0].label
    bbx_list = torch.from_numpy(np.array([[bbx.xmin, bbx.ymin, bbx.xmax, bbx.ymax] for bbx in detect_result_bbx_list]))
    bbx_list_conf = torch.from_numpy(
        np.array([[bbx.xmin, bbx.ymin, bbx.xmax, bbx.ymax, bbx.conf] for bbx in detect_result_bbx_list]))
    ious = box_iou(bbx_list, bbx_list)
    already_merged = set()  # 已遍历节点
    bbx_fusion_list = []
    while len(already_merged) < len(bbx_list):
        for i in range(len(bbx_list)):
            if i in already_merged:
                continue
            else:
                already_merged.add(i)
                queue = []  # 当前广度优先遍历层
                queue.append(i)
                bbx_cluster = [i]
                break
        while len(queue) > 0:
            bbx = queue.pop()
            for i in range(len(bbx_list)):
                if i in already_merged:
                    continue
                if ious[bbx, i] > 0.01:
                    already_merged.add(i)
                    queue = [i] + queue
                    bbx_cluster.append(i)

        def merge_bbx_cluster(bbx_cluster):
            bbx_cluster = torch.from_numpy(np.array(bbx_cluster))
            xmin = bbx_cluster[:, 0].min()
            ymin = bbx_cluster[:, 1].min()
            xmax = bbx_cluster[:, 2].max()
            ymax = bbx_cluster[:, 3].max()
            conf = bbx_cluster[:, 4].max()
            return proto_gen.detect_pb2.DetectResultBBX(xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax),
                                                        label=label, conf=conf)

        bbx_fusion_list.append(
            merge_bbx_cluster(bbx_list_conf[bbx_cluster])
        )
    return bbx_fusion_list


def nms_fusion(
        all_detect_result_bbx_list: List[proto_gen.detect_pb2.DetectResultBBX],
) -> List[proto_gen.detect_pb2.DetectResultBBX]:
    all_detect_result_bbx_list = all_detect_result_bbx_list[:]
    # step1. label -> detect_result_bbx_list
    labeled_detect_result_bbx_list_map = {}
    while len(all_detect_result_bbx_list) > 0:
        bbx = all_detect_result_bbx_list.pop(0)
        if bbx.label in labeled_detect_result_bbx_list_map:
            labeled_detect_result_bbx_list_map[bbx.label].append(bbx)
        else:
            labeled_detect_result_bbx_list_map[bbx.label] = [bbx]
    # step2. bbx cluster
    bbx_fusion_list = []
    for _, detect_result_bbx_list in labeled_detect_result_bbx_list_map.items():
        bbx_fusion_list.extend(
            nms_fusion_single_label(
                detect_result_bbx_list
            )
        )
    return bbx_fusion_list


if __name__ == '__main__':
    box1 = [0.12, 0.33, 0.72, 0.64]
    box2 = [0.10, 0.31, 0.71, 0.61]
    print(iou(box1, box2))

    box1 = [12, 33, 72, 64]
    box2 = [10, 31, 71, 61]
    print(iou(box1, box2))

    all_detect_result_bbx_list = [
        proto_gen.detect_pb2.DetectResultBBX(xmin=0, ymin=51, xmax=81, ymax=91, conf=0.9, label=0, ),
        proto_gen.detect_pb2.DetectResultBBX(xmin=10, ymin=31, xmax=71, ymax=61, conf=0.8, label=1, ),
        proto_gen.detect_pb2.DetectResultBBX(xmin=1, ymin=32, xmax=83, ymax=93, conf=0.2, label=0, ),
        proto_gen.detect_pb2.DetectResultBBX(xmin=2, ymin=53, xmax=11, ymax=94, conf=0.4, label=1, ),
        proto_gen.detect_pb2.DetectResultBBX(xmin=3, ymin=24, xmax=12, ymax=35, conf=0.7, label=1, ),
        proto_gen.detect_pb2.DetectResultBBX(xmin=4, ymin=56, xmax=84, ymax=92, conf=0.5, label=1, ),
        proto_gen.detect_pb2.DetectResultBBX(xmin=12, ymin=33, xmax=72, ymax=64, conf=0.8, label=1, ),
        proto_gen.detect_pb2.DetectResultBBX(xmin=38, ymin=66, xmax=79, ymax=95, conf=0.7, label=1, ),
        proto_gen.detect_pb2.DetectResultBBX(xmin=8, ymin=49, xmax=21, ymax=89, conf=0.3, label=0, ),
    ]
    # nms_fusion(all_detect_result_bbx_list)

    import cv2
    import utils.plot

    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse()
    yolo_model_resp.ParseFromString(open('./demo_nms_yolo_model_resp', 'rb').read())

    label_yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=yolo_model_resp.image_path,
        detect_result_bbx_list=[bbx for bbx in yolo_model_resp.detect_result_bbx_list if bbx.label == 2],
        ## 2 6 7 8 9 10
    )
    cv2.imshow('label_yolo_model_resp.png', utils.plot.PlotYolov5ModelResponse(label_yolo_model_resp)[:, ::])
    cv2.imwrite('nms_nmf/label_yolo_model_resp.png',
                utils.plot.PlotYolov5ModelResponse(label_yolo_model_resp)[200: 500, 420: 750, :])

    bbx_fusion_list = nms_fusion(label_yolo_model_resp.detect_result_bbx_list)
    fusion_yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=yolo_model_resp.image_path,
        detect_result_bbx_list=bbx_fusion_list,
    )
    cv2.imshow('fusion_yolo_model_resp.png', utils.plot.PlotYolov5ModelResponse(fusion_yolo_model_resp)[:, :, :])
    cv2.imwrite('nms_nmf/fusion_yolo_model_resp.png',
                utils.plot.PlotYolov5ModelResponse(fusion_yolo_model_resp)[200: 500, 420: 750, :])
    # 2: 462，287，699，407
    # 10：517，202，731，254
    # 9：736，217，802，316

    import detect.nms

    nms_list = detect.nms.non_max_suppression(label_yolo_model_resp.detect_result_bbx_list)
    nms_yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=yolo_model_resp.image_path,
        detect_result_bbx_list=nms_list,
    )
    cv2.imshow('nms_yolo_model_resp.png', utils.plot.PlotYolov5ModelResponse(nms_yolo_model_resp)[:, :, :])
    cv2.imwrite('nms_nmf/nms_yolo_model_resp.png',
                utils.plot.PlotYolov5ModelResponse(nms_yolo_model_resp)[200: 500, 420: 750, :])

    cv2.waitKey(0)
