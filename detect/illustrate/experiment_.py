import logging

import cv2

import proto_gen.detect_pb2
import dataset.format_dataset
import metrics_utils.mAP

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.ERROR,
)


def weight_proj_detect(proj_detect_func: callable, pano_weight_path: str, proj_weight_path: str):
    def detect(req: proto_gen.detect_pb2.YoloModelRequest):
        return proj_detect_func(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=pano_weight_path,
            ),
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_weight_path,
            ),
        )

    return detect


def sub_experiment(weight_detect_func: callable, exp_group: str, exp_name: str):
    format_dataset = dataset.format_dataset.SubFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    mAP = metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="",
        detect_func=weight_detect_func,
        detect_save_dir=f"/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/{exp_group}",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir=f'/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/experiments/{exp_name}',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )
    logging.error(f"{exp_name}: {mAP}")


if __name__ == '__main__':
    import detect.multi_project_detect
    import detect.multi_gen_project_detect
    import detect.mvpf_detect
    import detect.self_mvpf_detect
    import time
    import utils.plot

    format_dataset = dataset.format_dataset.SubFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        sub_index_list=[2, 14, 18, 20],
    )

    for i, data in enumerate(format_dataset):
        cv2.imshow("data", utils.plot.PlotDatasetModel(data))
        cv2.imwrite(f"/Users/bytedance/Desktop/data_{i}.png", utils.plot.PlotDatasetModel(data))

        curTime = time.time()
        resp = weight_proj_detect(
            detect.multi_gen_project_detect.multi_gen_project_detect,
            pano_weight_path='/weights/equi_1n_exp13_best.pt',
            proj_weight_path='/weights/stereo_1n_d_exp113_best.pt',
        )(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
        ))
        logging.error(f'{i} time cost {time.time() - curTime} s')

        cv2.imshow("resp", utils.plot.PlotDatasetModel(data))
        cv2.imwrite(f"/Users/bytedance/Desktop/resp_wo_{i}.png", utils.plot.PlotYolov5ModelResponse(resp))

        curTime = time.time()
        resp = weight_proj_detect(
            detect.self_mvpf_detect.self_mvpf_detect,
            pano_weight_path='/weights/equi_1n_exp13_best.pt',
            proj_weight_path='/weights/stereo_1n_d_exp113_best.pt',
        )(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
        ))
        logging.error(f'{i} time cost {time.time() - curTime} s')

        cv2.imshow("resp", utils.plot.PlotDatasetModel(data))
        cv2.imwrite(f"/Users/bytedance/Desktop/resp_w_{i}.png", utils.plot.PlotYolov5ModelResponse(resp))

        curTime = time.time()
        resp = weight_proj_detect(
            detect.multi_project_detect.multi_project_detect_2_only_project,
            pano_weight_path='/weights/equi_1n_exp13_best.pt',
            proj_weight_path='/weights/stereo_1n_d_exp113_best.pt',
        )(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
        ))
        logging.error(f'{i} time cost {time.time() - curTime} s')

        cv2.imshow("resp", utils.plot.PlotDatasetModel(data))
        cv2.imwrite(f"/Users/bytedance/Desktop/resp_gen_{i}.png", utils.plot.PlotYolov5ModelResponse(resp))

        if i == 8:
            exit(-1)
        # cv2.waitKey(0)

# sub_experiment(
#     weight_proj_detect(
#         detect.multi_gen_project_detect.multi_gen_project_detect_weighted,
#         pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/std_n_exp88_best.pt',
#         proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_std_exp10_best.pt',
#     ),
#     "multi_gen_project_detect_",
#     "sub_experiment_multi_gen_project_detect_weighted")
#
# sub_experiment(
#     weight_proj_detect(
#         detect.mvpf_detect.mvpf_detect,
#         pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/std_n_exp88_best.pt',
#         proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_std_exp10_best.pt',
#     ),
#     "mvpf_",
#     "sub_experiment_mvpf_detect_std_std")
#
# sub_experiment(
#     weight_proj_detect(
#         detect.mvpf_detect.mvpf_detect,
#         pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
#         proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_std_exp10_best.pt',
#     ),
#     "mvpf_",
#     "sub_experiment_mvpf_detect_equi_std")
#
# sub_experiment(
#     weight_proj_detect(
#         detect.mvpf_detect.mvpf_detect,
#         pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
#         proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_exp25_best.pt',
#     ),
#     "mvpf_",
#     "sub_experiment_mvpf_detect_equi_stereo")
#
# sub_experiment(
#     weight_proj_detect(
#         detect.mvpf_detect.mvpf_detect,
#         pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
#         proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_exp7_best.pt',
#     ),
#     "mvpf_",
#     "sub_experiment_mvpf_detect_equi_stereo_2")
#
# sub_experiment(
#     weight_proj_detect(
#         detect.mvpf_detect.mvpf_detect,
#         pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
#         proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
#     ),
#     "mvpf_",
#     "sub_experiment_mvpf_detect_equi_stereo_2_d")
