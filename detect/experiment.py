import logging

import proto_gen.detect_pb2
import client.object_detection_client
import dataset.format_dataset
import metrics_utils.mAP
import detect.project_detect
import detect.multi_project_detect
import detect.multi_gen_project_detect

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO,
)


def demo_proj_detect(req: proto_gen.detect_pb2.YoloModelRequest) -> proto_gen.detect_pb2.YoloModelResponse:
    return detect.project_detect.project_detect(
        req,
        proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=0.0))


def stereo_1n_d_exp113_best_mini_10_12():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=10,
    )

    def stereo_1n_d_exp113_best(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_project_detect.multi_project_detect(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
        detect_func=stereo_1n_d_exp113_best,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/stereo_1n_d_exp113_best_mini_10_12",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/bak/stereo_1n_d_exp113_best_mini_10_12',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


def multi_gen_stereo_1n_d_exp113_best_mini_10_12():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=1,
    )

    def stereo_1n_d_exp113_best(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
        detect_func=stereo_1n_d_exp113_best,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_stereo_1n_d_exp113_best_mini_10_12",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/bak/multi_gen_stereo_1n_d_exp113_best_mini_10_12',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )

def multi_gen_project_detect_weighted_exp113_best_mini_10_12():
    format_dataset = dataset.format_dataset.MiniFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        mini_size=10,
    )

    def stereo_1n_d_exp113_best(req: proto_gen.detect_pb2.YoloModelRequest):
        return detect.multi_gen_project_detect.multi_gen_project_detect_weighted(
            req=req,
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
            ),
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
        detect_func=stereo_1n_d_exp113_best,
        detect_save_dir="/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect_weighted_exp113_best_mini_10_12",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir='/experiments/bak/multi_gen_project_detect_weighted_exp113_best_mini_10_12',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )

if __name__ == '__main__':
    multi_gen_project_detect_weighted_exp113_best_mini_10_12()