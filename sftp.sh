#! /bin/bash

read -p "Do you want to put the file? (y/n) " answer

cd $(dirname $0)
./push.sh

sftp lab724 << EOF
put -r /Users/limengfan/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection /home/lmf/Deploy
EOF
