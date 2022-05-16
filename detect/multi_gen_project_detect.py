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


def multi_gen_project_detect(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
        iou_thr: float = 0.3,
) -> proto_gen.detect_pb2.YoloModelResponse:
    yolo_model_resp = detect_func(req)
    all_proj_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
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
        all_proj_resp.detect_result_bbx_list.extend(proj_resp.detect_result_bbx_list)
    ret_proj_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    ret_proj_resp.detect_result_bbx_list.extend(
        detect.nms.non_max_suppression(all_proj_resp.detect_result_bbx_list, iou_thr=iou_thr))
    return ret_proj_resp


def weighted_detect(req: proto_gen.detect_pb2.YoloModelRequest):
    return multi_gen_project_detect(
        req=proto_gen.detect_pb2.YoloModelRequest(
            image_path=req.image_path,
            weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
        ),
        proj_req=proto_gen.detect_pb2.YoloModelRequest(
            image_path=req.image_path,
            weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
        ),
    )


def multi_gen_project_detect_weighted(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_req: proto_gen.detect_pb2.YoloModelRequest,
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
        iou_thr: float = 0.3,
        ensemble_boxes_nms_func: callable = weighted_boxes_fusion,
) -> proto_gen.detect_pb2.YoloModelResponse:
    yolo_model_resp = detect_func(req)
    all_detect_result_bbx_list = []
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
        all_detect_result_bbx_list.append(proj_resp.detect_result_bbx_list)
    ret_proj_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    ret_proj_resp.detect_result_bbx_list.extend(
        detect.nms.weighted_nms_adapter(
            all_detect_result_bbx_list,
            iou_thr=iou_thr,
            ensemble_boxes_nms_func=ensemble_boxes_nms_func,
        ))
    return ret_proj_resp


# def multi_project_detect_stereo_1n(req: proto_gen.detect_pb2.YoloModelRequest):
#     return multi_project_detect(
#         req=req,
#         proj_req=proto_gen.detect_pb2.YoloModelRequest(
#             image_path=req.image_path,
#             weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
#         ),
#     )


if __name__ == '__main__':
    import utils.plot
    import dataset.format_dataset

    format_dataset = dataset.format_dataset.FormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    for i, data in enumerate(format_dataset):
        yolo_model_resp = client.object_detection_client.ObjectDetect(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
            weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt",
        ))
        cv2.imshow("equi_image", utils.plot.PlotDatasetModelAndYolov5ModelResponse(data, yolo_model_resp))

        yolo_model_resp = multi_gen_project_detect(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
            weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
        ), proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
            weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
        ))
        cv2.imshow("stereo_image", utils.plot.PlotDatasetModelAndYolov5ModelResponse(data, yolo_model_resp))

        yolo_model_resp = multi_gen_project_detect_weighted(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
            weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
        ), proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
            weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
        ))
        cv2.imshow("stereo_image_weighted", utils.plot.PlotDatasetModelAndYolov5ModelResponse(data, yolo_model_resp))
        cv2.imshow("stereo_image_weighted1", utils.plot.PlotYolov5ModelResponse(yolo_model_resp))

        cv2.waitKey(0)

        # yolo_model_req = proto_gen.detect_pb2.YoloModelRequest(
        #     image_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/demo/pano_0a46e210e7018af58de6f45f0997486c.png",
        #     weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt"
        # )
        #
        # # proj_req = proto_gen.detect_pb2.YoloModelRequest(
        # #     image_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/demo/pano_0a46e210e7018af58de6f45f0997486c.png",
        # #     weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_exp25_best.pt"
        # # )
        # # yolo_model_resp = multi_project_detect(
        # #     req=yolo_model_req,
        # #     proj_req=proj_req, )
        # # print(yolo_model_resp)
        # # cv2.imshow("image", utils.plot.PlotYolov5ModelResponse(yolo_model_resp))
        # # cv2.waitKey(0)
        #
        # yolo_model_resp = multi_project_detect_stereo_1n(
        #     req=yolo_model_req, )
        # print(yolo_model_resp)
        # cv2.imshow("image", utils.plot.PlotYolov5ModelResponse(yolo_model_resp))
        # cv2.waitKey(0)
