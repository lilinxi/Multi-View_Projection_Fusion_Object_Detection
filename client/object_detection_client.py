from __future__ import print_function

import logging
import platform

import grpc
import proto_gen.detect_pb2
import proto_gen.detect_pb2_grpc


def ObjectDetect(req: proto_gen.detect_pb2.YoloModelRequest) -> proto_gen.detect_pb2.YoloModelResponse:
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = proto_gen.detect_pb2_grpc.DeformYolov5Stub(channel)
        resp = stub.Detect(req)
    return resp


# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/greeter_client.py
def run():
    with grpc.insecure_channel('localhost:50052') as channel:
        stub = proto_gen.detect_pb2_grpc.DeformYolov5Stub(channel)

        sys = platform.system()
        if sys == 'Darwin':
            response = stub.Detect(
                proto_gen.detect_pb2.YoloModelRequest(
                    image_path="/Users/limengfan/PycharmProjects/DeformYolov5/data/images/data.png",
                    image_size=640,
                    weights_path="/Users/limengfan/PycharmProjects/DeformYolov5/yolov5n.pt",
                    conf_thres=0.25,
                    iou_thres=0.45,
                ),
            )
        elif sys == 'Linux':
            response = stub.Detect(
                proto_gen.detect_pb2.YoloModelRequest(
                    image_path="/home/lmf/Deploy/DeformYolov5/data/images/data.png",
                    image_size=640,
                    weights_path="/home/lmf/Deploy/DeformYolov5/yolov5n.pt",
                    conf_thres=0.25,
                    iou_thres=0.45,
                ),
            )
        else:
            print(f"RepF-Net do not support {sys} system")
            exit(255)

        # response = stub.Detect(
        #     proto_gen.detect_pb2.YoloModelRequest(
        #         image_path="/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/yolo/yolov5/data/images/bus.jpg",
        #         image_size=640,
        #         weights_path="/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_exp25_best.pt",
        #         # weights_path="/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/yolo/yolov5/yolov5n.pt",
        #         conf_thres=0.25,
        #         iou_thres=0.45,
        #     ),
        # )
    print("Greeter client received: ")
    print(response)


if __name__ == '__main__':
    logging.basicConfig()
    run()
