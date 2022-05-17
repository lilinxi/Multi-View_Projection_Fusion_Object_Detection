from typing import List

import cv2
import numpy as np

from ensemble_boxes import *

import client.object_detection_client
import proto_gen.detect_pb2
import proj.stereo_proj
import proj.perspective_proj
import proj.transform
import detect.project_detect
import detect.nms
import detect.nms_fusion


def self_mvpf_detect(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
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
    all_proj_resp.detect_result_bbx_list.extend(yolo_model_resp.detect_result_bbx_list)
    i = 0
    for detect_result_bbx in yolo_model_resp.detect_result_bbx_list:
        center = (detect_result_bbx.xmax + detect_result_bbx.xmin) / 2 / pano_width
        theta_rotate = center * np.pi * 2
        proj_resp = detect.project_detect.project_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path=proj_req.weights_path,
            ),
            proj_params=proto_gen.detect_pb2.StereoProjectParams(
                project_dis=1, project_size=2, theta_rotate=theta_rotate),
            proj_func=proj_func,
            projXY2panoXYZ_func=projXY2panoXYZ_func,
            detect_func=detect_func,
            proj_width=proj_width,
            proj_height=proj_height,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        import utils.plot
        image = cv2.imread(req.image_path)
        proj_image = proj_func(image, proto_gen.detect_pb2.StereoProjectParams(
                project_dis=1, project_size=2, theta_rotate=theta_rotate), proj_width=proj_width, proj_height=proj_height)

        cv2.imwrite(f'/Users/bytedance/Desktop/proj_resp_{i}.png', utils.plot.PlotYolov5ModelResponse(proj_resp))
        cv2.imwrite(f'/Users/bytedance/Desktop/proj_image_{i}.png', proj_image)
        print(f'{i},{theta_rotate}')
        i += 1
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)
    ret_proj_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    ret_proj_resp.detect_result_bbx_list.extend(
        detect.nms_fusion.nms_fusion(all_proj_resp.detect_result_bbx_list))
    return ret_proj_resp


def weighted_detect(req: proto_gen.detect_pb2.YoloModelRequest):
    return self_mvpf_detect(
        req=proto_gen.detect_pb2.YoloModelRequest(
            image_path=req.image_path,
            weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
        ),
        proj_req=proto_gen.detect_pb2.YoloModelRequest(
            image_path=req.image_path,
            weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
        ),
    )
