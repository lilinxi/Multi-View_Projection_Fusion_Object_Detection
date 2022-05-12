from typing import List

import cv2
import numpy as np

import client.object_detection_client
import proto_gen.detect_pb2
import proj.stereo_proj
import proj.perspective_proj
import proj.transform
import detect.project_detect
import detect.nms


def multi_project_detect_2_only_project(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    yolo_model_resp = detect_func(req)
    all_proj_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(detect.nms.non_max_suppression(all_proj_resp.detect_result_bbx_list))
    return all_proj_resp


def multi_project_detect_3_only_project(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    yolo_model_resp = detect_func(req)
    all_proj_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(detect.nms.non_max_suppression(all_proj_resp.detect_result_bbx_list))
    return all_proj_resp


def multi_project_detect_2_pano_nms_project(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    all_proj_resp = detect_func(req)
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(detect.nms.non_max_suppression(all_proj_resp.detect_result_bbx_list))
    return yolo_model_resp


def multi_project_detect_3_pano_nms_project(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    all_proj_resp = detect_func(req)
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(detect.nms.non_max_suppression(all_proj_resp.detect_result_bbx_list))
    return yolo_model_resp


def multi_project_detect_2_pano_nms_project_001(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    all_proj_resp = detect_func(req)
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(
        detect.nms.non_max_suppression(all_proj_resp.detect_result_bbx_list, iou_thr=0.001))
    return yolo_model_resp


def multi_project_detect_3_pano_nms_project_001(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    all_proj_resp = detect_func(req)
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(
        detect.nms.non_max_suppression(all_proj_resp.detect_result_bbx_list, iou_thr=0.001))
    return yolo_model_resp


def multi_project_detect_2_pano_weight_project(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    all_detect_result_bbx_list = []
    all_proj_resp = detect_func(req)
    all_detect_result_bbx_list.append(all_proj_resp.detect_result_bbx_list)
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_detect_result_bbx_list.append(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(detect.nms.weighted_nms(all_detect_result_bbx_list))
    return yolo_model_resp


def multi_project_detect_3_pano_weight_project(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    all_detect_result_bbx_list = []
    all_proj_resp = detect_func(req)
    all_detect_result_bbx_list.append(all_proj_resp.detect_result_bbx_list)
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_detect_result_bbx_list.append(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(detect.nms.weighted_nms(all_detect_result_bbx_list))
    return yolo_model_resp


def multi_project_detect_2_pano_weight_project_001(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    all_detect_result_bbx_list = []
    all_proj_resp = detect_func(req)
    all_detect_result_bbx_list.append(all_proj_resp.detect_result_bbx_list)
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_detect_result_bbx_list.append(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(detect.nms.weighted_nms(all_detect_result_bbx_list, iou_thr=0.01))
    return yolo_model_resp


def multi_project_detect_3_pano_weight_project_001(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params_list: List[proto_gen.detect_pb2.StereoProjectParams] = [
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=0.0),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi / 2),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=3, theta_rotate=np.pi * 3 / 2),
        ],
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    all_detect_result_bbx_list = []
    all_proj_resp = detect_func(req)
    all_detect_result_bbx_list.append(all_proj_resp.detect_result_bbx_list)
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    for proj_params in proj_params_list:
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proj_params,
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        all_detect_result_bbx_list.append(proj_resp.detect_result_bbx_list)
    yolo_model_resp.detect_result_bbx_list.extend(detect.nms.weighted_nms(all_detect_result_bbx_list, iou_thr=0.01))
    return yolo_model_resp

# if __name__ == '__main__':
#     import utils.plot
#     import dataset.format_dataset
#
#     format_dataset = dataset.format_dataset.FormatDataset(
#         dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
#         train=False,
#         image_width=1024,
#         image_height=512,
#         class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
#                       'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
#     )
#
#     for i, data in enumerate(format_dataset):
#         yolo_model_resp = client.object_detection_client.ObjectDetect(proto_gen.detect_pb2.YoloModelRequest(
#             image_path=data.image_path,
#             weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt",
#         ))
#         cv2.imshow("equi_image", utils.plot.PlotDatasetModelAndYolov5ModelResponse(data, yolo_model_resp))
#
#         yolo_model_resp = multi_project_detect(proto_gen.detect_pb2.YoloModelRequest(
#             image_path=data.image_path,
#             weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
#         ), proto_gen.detect_pb2.YoloModelRequest(
#             image_path=data.image_path,
#             weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
#         ))
#
#         cv2.imshow("stereo_image", utils.plot.PlotDatasetModelAndYolov5ModelResponse(data, yolo_model_resp))
#         cv2.waitKey(0)
#
#         # yolo_model_req = proto_gen.detect_pb2.YoloModelRequest(
#         #     image_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/demo/pano_0a46e210e7018af58de6f45f0997486c.png",
#         #     weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt"
#         # )
#         #
#         # # proj_req = proto_gen.detect_pb2.YoloModelRequest(
#         # #     image_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/demo/pano_0a46e210e7018af58de6f45f0997486c.png",
#         # #     weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_exp25_best.pt"
#         # # )
#         # # yolo_model_resp = multi_project_detect(
#         # #     req=yolo_model_req,
#         # #     proj_req=proj_req, )
#         # # print(yolo_model_resp)
#         # # cv2.imshow("image", utils.plot.PlotYolov5ModelResponse(yolo_model_resp))
#         # # cv2.waitKey(0)
#         #
#         # yolo_model_resp = multi_project_detect_stereo_1n(
#         #     req=yolo_model_req, )
#         # print(yolo_model_resp)
#         # cv2.imshow("image", utils.plot.PlotYolov5ModelResponse(yolo_model_resp))
#         # cv2.waitKey(0)
