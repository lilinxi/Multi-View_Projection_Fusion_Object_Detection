from typing import List

import numpy as np
import torch
import torchvision

from ensemble_boxes import *

import proto_gen.detect_pb2


def non_max_suppression(detect_result_bbx_list: List[proto_gen.detect_pb2.DetectResultBBX], iou_thr: float = 0.3) \
        -> List[proto_gen.detect_pb2.DetectResultBBX]:
    boxes = np.zeros((len(detect_result_bbx_list), 4), dtype=np.float32)
    scores = np.zeros((len(detect_result_bbx_list)), dtype=np.float32)
    for i, detect_result_bbx in enumerate(detect_result_bbx_list):
        boxes[i, :] = [detect_result_bbx.xmin, detect_result_bbx.ymin, detect_result_bbx.xmax, detect_result_bbx.ymax]
        scores[i] = detect_result_bbx.conf
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(scores)
    index = torchvision.ops.nms(boxes, scores, iou_thr)  # NMS
    return [detect_result_bbx_list[i] for i in index]


def weighted_nms_adapter(
        all_detect_result_bbx_list: List[List[proto_gen.detect_pb2.DetectResultBBX]],
        ensemble_boxes_nms_func: callable,
        iou_thr=0.5, weights: List[float] = None, **kwargs
) -> List[proto_gen.detect_pb2.DetectResultBBX]:
    """
    Modify ensemble_boxes
                elif not allows_overflow:
                # weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
                weighted_boxes[i, 1] = weighted_boxes[i, 1]
    """
    bbx_weight = 1024
    bbx_height = 512
    boxes_list = [
        np.zeros((len(detect_result_bbx_list), 4), dtype=np.float32)
        for detect_result_bbx_list in all_detect_result_bbx_list
    ]
    scores_list = [
        np.zeros((len(detect_result_bbx_list)), dtype=np.float32)
        for detect_result_bbx_list in all_detect_result_bbx_list
    ]
    labels_list = [
        np.zeros((len(detect_result_bbx_list)), dtype=np.int)
        for detect_result_bbx_list in all_detect_result_bbx_list
    ]
    for i, detect_result_bbx_list in enumerate(all_detect_result_bbx_list):
        for j, detect_result_bbx in enumerate(detect_result_bbx_list):
            boxes_list[i][j, :] = [
                detect_result_bbx.xmin / bbx_weight,
                detect_result_bbx.ymin / bbx_height,
                detect_result_bbx.xmax / bbx_weight,
                detect_result_bbx.ymax / bbx_height,
            ]
            scores_list[i][j] = detect_result_bbx.conf
            labels_list[i][j] = detect_result_bbx.label
    boxes_list = [boxes.tolist() for boxes in boxes_list]
    scores_list = [scores.tolist() for scores in scores_list]
    labels_list = [labels.tolist() for labels in labels_list]
    boxes, scores, labels = ensemble_boxes_nms_func(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thr, weights=weights, **kwargs)
    ret_detect_result_bbx_list = []
    for i in range(len(boxes)):
        ret_detect_result_bbx_list.append(
            proto_gen.detect_pb2.DetectResultBBX(
                xmin=round(boxes[i, 0] * bbx_weight),
                ymin=round(boxes[i, 1] * bbx_height),
                xmax=round(boxes[i, 2] * bbx_weight),
                ymax=round(boxes[i, 3] * bbx_height),
                label=round(labels[i]),
                conf=scores[i],
            )
        )

    return ret_detect_result_bbx_list


def weighted_nms(
        all_detect_result_bbx_list: List[List[proto_gen.detect_pb2.DetectResultBBX]],
        iou_thr: float = 0.3, weights: List[float] = None, **kwargs) \
        -> List[proto_gen.detect_pb2.DetectResultBBX]:
    return weighted_nms_adapter(
        all_detect_result_bbx_list, ensemble_boxes_nms_func=weighted_boxes_fusion,
        iou_thr=iou_thr,
        weights=weights, **kwargs)


def nms(
        all_detect_result_bbx_list: List[List[proto_gen.detect_pb2.DetectResultBBX]],
        iou_thr: float = 0.3, weights: List[float] = None, **kwargs) \
        -> List[proto_gen.detect_pb2.DetectResultBBX]:
    return weighted_nms_adapter(
        all_detect_result_bbx_list, ensemble_boxes_nms_func=nms,
        iou_thr=iou_thr,
        weights=weights, **kwargs)


def soft_nms(
        all_detect_result_bbx_list: List[List[proto_gen.detect_pb2.DetectResultBBX]],
        iou_thr: float = 0.3, weights: List[float] = None, **kwargs) \
        -> List[proto_gen.detect_pb2.DetectResultBBX]:
    return weighted_nms_adapter(
        all_detect_result_bbx_list, ensemble_boxes_nms_func=soft_nms,
        iou_thr=iou_thr,
        weights=weights, **kwargs)


def non_maximum_weighted(
        all_detect_result_bbx_list: List[List[proto_gen.detect_pb2.DetectResultBBX]],
        iou_thr: float = 0.3, weights: List[float] = None, **kwargs) \
        -> List[proto_gen.detect_pb2.DetectResultBBX]:
    return weighted_nms_adapter(
        all_detect_result_bbx_list, ensemble_boxes_nms_func=non_maximum_weighted,
        iou_thr=iou_thr,
        weights=weights, **kwargs)


if __name__ == '__main__':
    from ensemble_boxes import *

    boxes_list = [[
        # [0.00, 0.51, 0.81, 0.91],
        [0.10, 0.31, 0.71, 0.61],
        # [0.01, 0.32, 0.83, 0.93],
        # [0.02, 0.53, 0.11, 0.94],
        # [0.03, 0.24, 0.12, 0.35],
    ], [
        # [0.04, 0.56, 0.84, 0.92],
        [0.12, 0.33, 0.72, 0.64],
        # [0.38, 0.66, 0.79, 0.95],
        # [0.08, 0.49, 0.21, 0.89],
    ]]
    scores_list = [[0.9], [0.5]]
    labels_list = [[1], [1]]
    weights = [2, 1]

    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1

    # boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    # boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr,
    #                                  sigma=sigma, thresh=skip_box_thr)
    # boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr,
    #                                              skip_box_thr=skip_box_thr)
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    print(boxes, scores, labels)
