from typing import List
import os
import platform

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
import platform
import logging
from utils.logger import write_log


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
    # 日志
    work_dir = f'{req.image_path}.dir' # image save work dir
    log_path = f'{req.image_path}.log'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if not os.path.exists(log_path):
        open(log_path, 'w').close()
    write_log(log_path, req.image_path)
    # 算法
    yolo_model_resp = detect_func(req)
    all_proj_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    all_proj_resp.detect_result_bbx_list.extend(yolo_model_resp.detect_result_bbx_list)
    logging.info("self_mvpf_detect, self detect finish.")
    write_log(log_path, f'{len(yolo_model_resp.detect_result_bbx_list)}') # 日志
    i = 0
    for detect_result_bbx in yolo_model_resp.detect_result_bbx_list:
        center = (detect_result_bbx.xmax + detect_result_bbx.xmin) / 2 / pano_width
        theta_rotate = (center * np.pi * 2 + np.pi) % (np.pi * 2)
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
            project_index=i,
            work_dir=work_dir,
        )

        import utils.plot
        image = cv2.imread(req.image_path)
        proj_image = proj_func(image, proto_gen.detect_pb2.StereoProjectParams(
            project_dis=1, project_size=2, theta_rotate=theta_rotate), proj_width=proj_width, proj_height=proj_height)

        # 投影图像
        cv2.imwrite(f'{work_dir}/proj_image_{i}.png', proj_image)
        # 重投影图像 + 单个检测框
        cv2.imwrite(f'{work_dir}/re_proj_image{i}.png',
                    utils.plot.PlotDetectBBX(proj_resp.image_path, detect_result_bbx))
        # 重投影图像 + 重投影检测框
        cv2.imwrite(f'{work_dir}/re_proj_image_result_{i}.png',
                    utils.plot.PlotYolov5ModelResponse(proj_resp))
        print(f'{i},{theta_rotate}')

        # 日志
        write_log(log_path, f'{work_dir}/proj_image_{i}.png')

        logging.info(f"self_mvpf_detect, mvp detect finish. {i}")
        i += 1
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)

    # 日志
    for ii in range(i):
        write_log(log_path, f'{work_dir}/proj_result_{ii}.png')
    for ii in range(i):
        write_log(log_path, f'{work_dir}/re_proj_image{ii}.png')
    for ii in range(i):
        write_log(log_path, f'{work_dir}/re_proj_image_result_{ii}.png')

    ret_proj_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    ret_proj_resp.detect_result_bbx_list.extend(
        detect.nms_fusion.nms_fusion(all_proj_resp.detect_result_bbx_list))

    import utils.plot
    # 几何先验
    cv2.imwrite(f'{work_dir}/yolo_model_resp.png',
                utils.plot.PlotYolov5ModelResponse(yolo_model_resp))
    # NMF 之前
    cv2.imwrite(f'{work_dir}/all_proj_resp.png',
                utils.plot.PlotYolov5ModelResponse(all_proj_resp))
    # NMF 之后
    cv2.imwrite(f'{work_dir}/ret_proj_resp.png',
                utils.plot.PlotYolov5ModelResponse(ret_proj_resp))

    write_log(log_path, f'{work_dir}/ret_proj_resp.png')

    return ret_proj_resp


def weighted_detect(req: proto_gen.detect_pb2.YoloModelRequest):
    sys = platform.system()
    if sys == 'Darwin':
        return self_mvpf_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/Users/limengfan/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
            ),
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/Users/limengfan/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
            ),
        )
    elif sys == 'Linux':
        return self_mvpf_detect(
            req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/home/lmf/Deploy/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
            ),
            proj_req=proto_gen.detect_pb2.YoloModelRequest(
                image_path=req.image_path,
                weights_path='/home/lmf/Deploy/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
            ),
        )
    else:
        print(f"RepF-Net Server do not support {sys} system")
        exit(255)
