# Multi-View_Projection_Fusion_Object_Detection

## install

```shell script
/usr/local/bin/python3 -m pip install opencv-python 
/usr/local/bin/python3 -m pip install scipy 
/usr/local/bin/python3 -m pip install torch
/usr/local/bin/python3 -m pip install matplotlib
/usr/local/bin/python3 -m pip install seaborn
/usr/local/bin/python3 -m pip install pyyaml
/usr/local/bin/python3 -m pip install ensemble-boxes
/usr/local/bin/python3 -m pip install open3d
```

## 目录介绍

- client：客户端
    - object_detection_client.py：目标检测客户端
- dataset：数据集生成
    - sun360_extended_dataset：读取原始数据集
    - format_dataset：读取格式化数据集
    - projection_sun360_extended_dataset：投影原始数据集
- demo：示例
- detect：目标检测
    - mvpf_detect：MVPF 目标检测
    - nms：非极大值抑制
    - project_detect：投影目标检测
- metrics_utils：评价指标
    - mAP：计算平均精度
    - metrics_utils：评价指标工具
- proj：投影
    - perspective_proj：透视投影
    - proj_utils：投影工具
    - stereo_conv_utils：立体卷积工具
    - stereo_proj：立体投影
    - transform：坐标转换工具
- proto：接口协议
- proto_gen：接口协议生成代码
- utils：工具函数
    - nda2bytes：numpy数组转bytes
    - plot：绘图工具

## 数据集

1. sun360_extended_dataset：原始数据集
2. sun360_extended_dataset_delete_error：原始数据集中删除错误的图片
3. sun360_extended_dataset_format：原始数据集删除错误标注后格式化