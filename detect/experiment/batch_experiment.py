import logging

import proto_gen.detect_pb2
import client.object_detection_client
import dataset.format_dataset
import metrics_utils.mAP
import detect.project_detect
import detect.multi_project_detect
import detect.multi_gen_project_detect

from ensemble_boxes import *

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO,
)


def multi_gen_project_detect_mini_10():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=10,
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_mini_10",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_mini_10',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


def multi_gen_project_detect():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_gen_project_detect_mini_10_001():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=10,
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
            iou_thr=0.001,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_mini_10_001",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_mini_10_001',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


def multi_gen_project_detect_001():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
            iou_thr=0.001,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_001",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_001',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_gen_project_detect_weighted_mini_10():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=10,
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_mini_10",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_weighted_mini_10',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


def multi_gen_project_detect_weighted_detect():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_detect",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_weighted_detect',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_gen_project_detect_weighted_mini_10_001():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=10,
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
            iou_thr=0.001,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_mini_10_001",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_weighted_mini_10_001',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


def multi_gen_project_detect_weighted_001():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
            iou_thr=0.001,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_001",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_weighted_001',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_gen_project_detect_weighted_mini_10_nms():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=10,
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
            ensemble_boxes_nms_func=nms,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_mini_10_nms",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_weighted_mini_10_nms',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


def multi_gen_project_detect_weighted_detect_nms():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
            ensemble_boxes_nms_func=nms,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_detect_nms",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_weighted_detect_nms',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_gen_project_detect_weighted_mini_10_001_nms():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=10,
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
            iou_thr=0.001,
            ensemble_boxes_nms_func=nms,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_mini_10_001_nms",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_weighted_mini_10_001_nms',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


def multi_gen_project_detect_weighted_001_nms():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
            iou_thr=0.001,
            ensemble_boxes_nms_func=nms,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_001_nms",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_gen_project_detect_weighted_001_nms',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_2_only_project():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_2_only_project(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_2_only_project",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_2_only_project',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_3_only_project():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_3_only_project(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_3_only_project",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_3_only_project',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_2_pano_nms_project():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_2_pano_nms_project(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_2_pano_nms_project",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_2_pano_nms_project',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_3_pano_nms_project():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_3_pano_nms_project(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_3_pano_nms_project",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_3_pano_nms_project',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_2_pano_nms_project_001():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_2_pano_nms_project_001(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_2_pano_nms_project_001",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_2_pano_nms_project_001',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_3_pano_nms_project_001():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_3_pano_nms_project_001(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_3_pano_nms_project_001",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_3_pano_nms_project_001',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_2_pano_weight_project():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_2_pano_weight_project(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_2_pano_weight_project",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_2_pano_weight_project',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_3_pano_weight_project():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_3_pano_weight_project(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_3_pano_weight_project",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_3_pano_weight_project',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_2_pano_weight_project_001():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_2_pano_weight_project_001(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_2_pano_weight_project_001",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_2_pano_weight_project_001',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


def multi_project_detect_3_pano_weight_project_001():
    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    def multi_gen_project_detect_mini_10(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect_3_pano_weight_project_001(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/weights/stereo_1n_d_exp113_best.pt",
        detect_func=multi_gen_project_detect_mini_10,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_project_detect_3_pano_weight_project_001",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/batch/multi_project_detect_3_pano_weight_project_001',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},
    )


if __name__ == '__main__':
    multi_gen_project_detect_mini_10()  # 0.481
    multi_gen_project_detect()  # 0.439
    multi_gen_project_detect_mini_10_001()  # 0.353
    multi_gen_project_detect_001()  # 0.398
    multi_gen_project_detect_weighted_mini_10()  # 0.481
    multi_gen_project_detect_weighted_detect()  # 0.472
    multi_gen_project_detect_weighted_mini_10_001()  # 0.508 ##########
    multi_gen_project_detect_weighted_001()  # 0.470
    multi_gen_project_detect_weighted_mini_10_nms()  # 0.488
    multi_gen_project_detect_weighted_detect_nms()  # Error
    multi_gen_project_detect_weighted_mini_10_001_nms()  # 0.466
    multi_gen_project_detect_weighted_001_nms()  # Error
    multi_project_detect_2_only_project()  # 0.474 ##########
    multi_project_detect_3_only_project()  # 0.358
    multi_project_detect_2_pano_nms_project()  # 0.573 ##########
    multi_project_detect_3_pano_nms_project()  # 0.514
    multi_project_detect_2_pano_nms_project_001()  # 0.462
    multi_project_detect_3_pano_nms_project_001()  # 0.420
    multi_project_detect_2_pano_weight_project()  # 0.541
    multi_project_detect_3_pano_weight_project()  # 0.517
    multi_project_detect_2_pano_weight_project_001()  # 0.526
    multi_project_detect_3_pano_weight_project_001()  # 0.524
