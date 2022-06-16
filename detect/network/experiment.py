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
    import utils.plot

    format_dataset = dataset.format_dataset.SubFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        sub_index_list=[13],
    )

    for i, data in enumerate(format_dataset):
        cv2.imshow(f"{i}_data", utils.plot.PlotDatasetModel(data))
        cv2.imwrite(f"/Users/bytedance/Desktop/data.png", utils.plot.PlotDatasetModel(data, plot_bbx=False))
        cv2.imwrite(f"/Users/bytedance/Desktop/data_gt.png", utils.plot.PlotDatasetModel(data, plot_bbx=True))

        resp = weight_proj_detect(
            detect.self_mvpf_detect.self_mvpf_detect,
            pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
            proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
        )(proto_gen.detect_pb2.YoloModelRequest(
            image_path=data.image_path,
        ))

        cv2.imshow("resp", utils.plot.PlotDatasetModel(data))
        cv2.imwrite(f"/Users/bytedance/Desktop/resp.png", utils.plot.PlotYolov5ModelResponse(resp))

        cv2.waitKey(0)
