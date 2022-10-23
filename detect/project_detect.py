from typing import List

import os
import cv2
import hashlib

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
    # 处理投影图没有跨边界，但是反投影跨边界的情况，一个投影检测在全景中可能分成两个，如果太小也可能一个都没有
    pano_bbx_list = []
    if minXY[0] - maxXY[0] > 2:
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
    elif maxXY[0] - minXY[0] > 2:
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
        project_index: int = 0,
        work_dir: str = None,
) -> proto_gen.detect_pb2.YoloModelResponse:
    yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse(
        image_path=req.image_path,
        detect_result_bbx_list=[],
    )
    image = cv2.imread(req.image_path)

    req_str = proj_params.SerializeToString()
    req_md5 = hashlib.md5(req_str).hexdigest()
    with open(req.image_path, 'rb') as fp:
        data = fp.read()
    file_md5 = hashlib.md5(data).hexdigest()
    hash_req = f'{req_md5}_{file_md5}'
    if not os.path.exists('/tmp/project_detect'):
        os.makedirs('/tmp/project_detect')
    cache_file = f'/tmp/project_detect/{hash_req}.png'

    if not os.path.exists(cache_file):
        proj_image = proj_func(image, proj_params, proj_width=proj_width, proj_height=proj_height)
        cv2.imwrite(cache_file, proj_image)

    proj_yolo_model_resp = detect_func(
        proto_gen.detect_pb2.YoloModelRequest(
            image_path=cache_file,
            image_size=req.image_size,
            weights_path=req.weights_path,
            conf_thres=req.conf_thres,
            iou_thres=req.iou_thres,
        ))

    import utils.plot
    cv2.imwrite(f'{work_dir}/proj_result_{project_index}.png',
                utils.plot.PlotYolov5ModelResponse(proj_yolo_model_resp))

    for bbx in proj_yolo_model_resp.detect_result_bbx_list:
        yolo_model_resp.detect_result_bbx_list.extend(
            detect_box_project2pano(bbx, proj_params, projXY2panoXYZ_func=projXY2panoXYZ_func,
                                    proj_width=proj_width, proj_height=proj_height,
                                    pano_width=pano_width, pano_height=pano_height)
        )

    # cv2.imwrite(f'/Users/bytedance/Desktop/re-proj_resp_{project_index}.png',
    #             utils.plot.PlotYolov5ModelResponse(yolo_model_resp))
    return yolo_model_resp


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
        cv2.imshow("data", utils.plot.PlotDatasetModel(data))

        yolo_model_resp = client.object_detection_client.ObjectDetect(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
            weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt",
        ))
        cv2.imshow("equi_image", utils.plot.PlotYolov5ModelResponse(yolo_model_resp))

        yolo_model_resp = project_detect(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
            weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt",
        ),
            proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2, theta_rotate=4.319689898685965))
        cv2.imshow("stereo_image", utils.plot.PlotYolov5ModelResponse(yolo_model_resp))

        cv2.waitKey(0)
