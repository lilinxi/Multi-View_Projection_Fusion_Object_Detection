import os
import shutil
import logging

import torch
import numpy as np

import proto_gen.detect_pb2

import dataset.format_dataset
import metrics_utils.yolov5_metrics_utils


def compute_mAP(
        format_dataset: dataset.format_dataset.FormatDataset,
        weights_path: str,
        detect_func: callable,
        detect_save_dir: str,
        cache: bool,
        names_dict: dict,
        plot_save_dir: str,
        plot_names_dict: dict = None,
        device: str = '',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
) -> float:
    """
    Compute mAP for a given dataset.
    """
    # 如果不缓存就删除上次的检测结果
    if not cache and os.path.exists(detect_save_dir):
        shutil.rmtree(detect_save_dir)
    # 创建文件夹存储检测结果
    os.path.exists(detect_save_dir) or os.makedirs(detect_save_dir)
    os.path.exists(plot_save_dir) or os.makedirs(plot_save_dir)

    device = metrics_utils.yolov5_metrics_utils.select_device(device)
    if plot_names_dict:
        names_dict = {k: v for k, v in names_dict.items() if plot_names_dict[k]}
        names_dict_map = {key: index for index, key in enumerate(names_dict)}  # label 映射，新 label dict 的字典序下标作为新 label
        names_dict = {names_dict_map[k]: v for k, v in names_dict.items()}  # 新 label dict
    # iou 步长划分
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    # 存储 Metrics
    stats = []
    confusion_matrix = metrics_utils.yolov5_metrics_utils.ConfusionMatrix(nc=len(names_dict))
    for i, data in enumerate(format_dataset):
        logging.info(f'Processing {i}th image...')
        # step1. 读取数据标注
        tcls = []
        labelsn = np.zeros((len(data.ground_truth_bbx_list), 5), dtype=np.float32)
        for i, bbx in enumerate(data.ground_truth_bbx_list):
            if plot_names_dict:
                if not plot_names_dict[bbx.label]:  # 不评估此 label 跳过
                    continue
                else:  # 评估此 label，但需要映射
                    bbx.label = names_dict_map[bbx.label]
            labelsn[i] = [bbx.label, bbx.xmin, bbx.ymin, bbx.xmax, bbx.ymax]
            tcls.append(bbx.label)
        labelsn = torch.from_numpy(labelsn).float().to(device)
        # step2. 获取检测结果并缓存
        detect_result_file = f'{detect_save_dir}/{data.image_filename}.txt'
        if os.path.exists(detect_result_file):
            logging.info(f'{detect_result_file} exists, skip...')
            yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse()
            yolo_model_resp.ParseFromString(open(detect_result_file, 'rb').read())
        else:
            yolo_model_resp = detect_func(proto_gen.detect_pb2.YoloModelRequest(
                image_path=data.image_path,
                weights_path=weights_path,
            ))
            with open(detect_result_file, 'wb') as f:
                f.write(yolo_model_resp.SerializeToString())
        # step3. 展示检测结果
        # cv2.imshow('image', dataset.plot_utils.PlotDatasetModelAndDetectResp(
        #     data, yolo_model_resp,
        #     ['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
        #      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        # ))
        # cv2.waitKey(0)
        # step4. 解析检测结果
        predn = np.zeros((len(yolo_model_resp.detect_result_bbx_list), 6), dtype=np.float32)
        for i, bbx in enumerate(yolo_model_resp.detect_result_bbx_list):
            if plot_names_dict:
                if not plot_names_dict[bbx.label]:  # 不评估此 label 跳过
                    continue
                else:  # 评估此 label，但需要映射
                    bbx.label = names_dict_map[bbx.label]
            predn[i] = [bbx.xmin, bbx.ymin, bbx.xmax, bbx.ymax, bbx.conf, bbx.label]
        predn = torch.from_numpy(predn).float().to(device)
        # step5. 计算检测指标
        correct = metrics_utils.yolov5_metrics_utils.process_batch(predn, labelsn, iouv)
        stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
        confusion_matrix.process_batch(predn, labelsn)

    # step6. 绘制检测指标
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, unique_classes = \
            metrics_utils.yolov5_metrics_utils.ap_per_class(*stats, plot=True, save_dir=plot_save_dir, names=names_dict)
        mAP = ap[:, 0].mean()
    confusion_matrix.plot(save_dir=plot_save_dir, names=list(names_dict.values()))
    logging.info(f'save: {plot_save_dir}, {detect_save_dir}')
    return mAP


if __name__ == '__main__':
    import client.object_detection_client

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.ERROR,
    )

    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    compute_mAP(
        format_dataset=format_dataset,
        weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt",
        detect_func=client.object_detection_client.ObjectDetect,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/equi_1n_exp13_best",
        cache=True,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/bak/equi_1n_exp13_best',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )
