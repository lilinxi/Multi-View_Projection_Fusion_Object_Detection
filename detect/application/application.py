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


if __name__ == '__main__':
    import detect.self_mvpf_detect
    import client.object_detection_client
    import time
    import utils.plot

    image_list = [
        'sun360_pano_aadmrasnwpvqnp',
        'sun360_pano_4ee978a83c6d02eb2bee454da5569011',
        'sfd2d3d_pano_4d28fa2a55f9a72dc619fa32cd29f327',
        'sfd2d3d_pano_0019e0a0c8ca0913e543c033a843c58f',
        'iGibson_input2',
        'iGibson_input5',
    ]

    for i, image_name in enumerate(image_list):
        image_path = f'/Users/bytedance/Desktop/DeepResult/dp_mvpf_mix/visualization/{image_name}/rgb.png'


        curTime = time.time()
        resp = client.object_detection_client.ObjectDetect(proto_gen.detect_pb2.YoloModelRequest(
            image_path=image_path,
            weights_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
        ))
        logging.error(f'{i} time cost {time.time() - curTime} s')

        cv2.imwrite(f"/Users/bytedance/Desktop/detect_{image_name}.png", utils.plot.PlotYolov5ModelResponse(resp))
