#!/usr/bin/env bash


cd `dirname $0` || exit 255
rm -rf proto_gen
mkdir proto_gen

#/usr/local/bin/python3 -m pip install grpcio
#/usr/local/bin/python3 -m pip install grpcio-tools
/usr/local/bin/python3 -m grpc_tools.protoc -I ./proto --python_out=proto_gen --grpc_python_out=proto_gen detect.proto
#rewrite
#from import detect_pb2 as detect__pb2
#to import proto_gen.detect_pb2 as detect__pb2


#brew install protobuf
#protoc --proto_path=./proto --python_out=./proto_gen detect.proto