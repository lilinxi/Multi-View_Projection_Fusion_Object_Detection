from typing import List

import cv2

import client.object_detection_client
import proto_gen.detect_pb2
import proj.stereo_proj
import proj.perspective_proj
import proj.transform


def detect_box_project2pano(bbx: proto_gen.detect_pb2.DetectResultBBX,
                            proj_params: proto_gen.detect_pb2.StereoProjectParams,
                            projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
                            proj_width: int = 640,
                            proj_height: int = 640,
                            pano_width: int = 1024,
                            pano_height: int = 512,
                            ) -> List[proto_gen.detect_pb2.DetectResultBBX]:
    min_xyz = projXY2panoXYZ_func(
        bbx.xmin / proj_width * proj_params.project_size,
        bbx.ymin / proj_height * proj_params.project_size, proj_params)
    minXY = proj.transform.xyz2XY(
        min_xyz[0], min_xyz[1], min_xyz[2],
        width=pano_width, height=pano_height,
        theta_rotate=proj_params.theta_rotate,
    )
    max_xyz = projXY2panoXYZ_func(
        bbx.xmax / proj_width * proj_params.project_size,
        bbx.ymax / proj_height * proj_params.project_size, proj_params)
    maxXY = proj.transform.xyz2XY(
        max_xyz[0], max_xyz[1], max_xyz[2],
        width=pano_width, height=pano_height,
        theta_rotate=proj_params.theta_rotate,
    )
    # 处理投影图没有跨边界，但是反投影跨边界的情况，一个投影检测在全景中可能分成两个
    if maxXY[0] < minXY[0]:
        pano_bbx_list = [
            proto_gen.detect_pb2.DetectResultBBX(
                xmin=round(minXY[0]),
                ymin=round(minXY[1]),
                xmax=pano_width,
                ymax=round(maxXY[1]),
                label=bbx.label,
                conf=bbx.conf,
            ),
            proto_gen.detect_pb2.DetectResultBBX(
                xmin=0,
                ymin=round(minXY[1]),
                xmax=round(maxXY[0]),
                ymax=round(maxXY[1]),
                label=bbx.label,
                conf=bbx.conf,
            ),
        ]
    else:
        pano_bbx_list = [proto_gen.detect_pb2.DetectResultBBX(
            xmin=round(minXY[0]),
            ymin=round(minXY[1]),
            xmax=round(maxXY[0]),
            ymax=round(maxXY[1]),
            label=bbx.label,
            conf=bbx.conf,
        )]
    return pano_bbx_list


def project_detect(
        req: proto_gen.detect_pb2.YoloModelRequest,
        proj_params: proto_gen.detect_pb2.StereoProjectParams,
        proj_func: callable = proj.stereo_proj.stereo_proj,
        projXY2panoXYZ_func: callable = proj.stereo_proj._projXY2panoXYZ,
        detect_func: callable = client.object_detection_client.ObjectDetect,
        proj_width: int = 640,
        proj_height: int = 640,
        pano_width: int = 1024,
        pano_height: int = 512,
) -> proto_gen.detect_pb2.YoloModelResponse:
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    image = cv2.imread(req.image_path)
    proj_image = proj_func(image, proj_params, proj_width=proj_width, proj_height=proj_height)
    cv2.imwrite('/tmp/project_detect/proj_image.jpg', proj_image)
    proj_yolo_model_resp = detect_func(
        proto_gen.detect_pb2.YoloModelRequest(
            image_path="/tmp/project_detect/proj_image.jpg",
            image_size=req.image_size,
            weights_path=req.weights_path,
            conf_thres=req.conf_thres,
            iou_thres=req.iou_thres,
        ))
    for bbx in proj_yolo_model_resp.detect_result_bbx_list:
        yolo_model_resp.detect_result_bbx_list.extend(
            detect_box_project2pano(bbx, proj_params, projXY2panoXYZ_func=projXY2panoXYZ_func,
                                    proj_width=proj_width, proj_height=proj_height,
                                    pano_width=pano_width, pano_height=pano_height)
        )
    return yolo_model_resp


# def stereo_detect(req: proto_gen.detect_pb2.YoloModelRequest) -> proto_gen.detect_pb2.YoloModelResponse:
#     """
#     This function is used to detect objects in a stereo image.
#     """
#     image = cv2.imread(req.image_path)
#     proj_params_list = [
#         proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=1, theta_rotate=0.0),
#         proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=1, theta_rotate=np.pi / 4),
#         proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=1, theta_rotate=np.pi / 2),
#         proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=1, theta_rotate=np.pi * 3 / 4),
#         proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=1, theta_rotate=np.pi),
#         proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=1, theta_rotate=np.pi * 5 / 4),
#         proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=1, theta_rotate=np.pi * 3 / 2),
#         proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=1, theta_rotate=np.pi * 7 / 4),
#     ]
#     yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
#         image_path=req.image_path,
#         detect_result_bbx_list=[],
#     )
#     for i, proj_params in enumerate(proj_params_list):
#         proj_image = proj.perspective_proj.perspective_proj(image, proj_params, 640, 640)
#         cv2.imwrite('/tmp/stereo_detect/proj_image.jpg', proj_image)
#         proj_yolo_model_resp = yolo.yolov5_service.Yolov5Detect(
#             proto_gen.detect_pb2.YoloModelRequest(
#                 image_path="/tmp/stereo_detect/proj_image.jpg",
#                 weights_path="/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/yolo/persp_83_best.pt"
#             ))
#         cv2.imshow(f'proj_image_{i}', yolo.yolov5_service.WarpYolov5ModelResponse(proj_yolo_model_resp))
#         for bbx in proj_yolo_model_resp.detect_result_bbx_list:
#             yolo_model_resp.detect_result_bbx_list.extend(
#                 stereo2pano(bbx, proj_params)
#             )
#     return yolo_model_resp


if __name__ == '__main__':
    import utils.plot

    yolo_model_req = proto_gen.detect_pb2.YoloModelRequest(
        image_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/demo/pano_0a46e210e7018af58de6f45f0997486c.png",
        weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_exp25_best.pt"
    )
    yolo_model_resp = project_detect(
        yolo_model_req,
        proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=0.0))
    print(yolo_model_resp)
    cv2.imshow("image", utils.plot.PlotYolov5ModelResponse(yolo_model_resp))
    cv2.waitKey(0)
