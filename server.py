from concurrent import futures
import logging

import grpc

import proto_gen.detect_pb2
import proto_gen.detect_pb2_grpc

import detect.self_mvpf_detect
import detect.multi_gen_project_detect


# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/greeter_server.py
class DeformYolov5Server(proto_gen.detect_pb2_grpc.DeformYolov5Servicer):
    def Detect(
            self,
            request: proto_gen.detect_pb2.YoloModelRequest,
            context
    ) -> proto_gen.detect_pb2.YoloModelResponse:
        logging.info(f'request: {request}')
        response = detect.self_mvpf_detect.weighted_detect(request)
        # response = detect.multi_gen_project_detect.weighted_detect(request)
        logging.info(f'response: {response}')
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto_gen.detect_pb2_grpc.add_DeformYolov5Servicer_to_server(DeformYolov5Server(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO,
    )
    logging.info("RepF-Net waiting at 50052")
    serve()
