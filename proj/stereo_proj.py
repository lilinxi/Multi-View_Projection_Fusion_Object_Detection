import math
import logging

import pickle

import cv2
from scipy import ndimage
import numpy as np

import proto_gen.detect_pb2

import proj.transform
import proj.proj_utils

StereoProjCache = {}


def _gen_frame_XY(width_size, height_size, width_shape, height_shape) -> np.ndarray:
    """
    采样二维矩阵
    Args:
        width_size: 采样的宽度大小
        height_size: 采样的高度大小
        width_shape: 采样的宽度维度
        height_shape: 采样的高度维度

    Returns: 采样二维矩阵，（height, width, [X, Y]）

    """
    X = np.linspace(0, width_size, width_shape) \
        .repeat(height_shape) \
        .reshape((width_shape, height_shape))
    Y = np.linspace(0, height_size, height_shape) \
        .repeat(width_shape) \
        .reshape((height_shape, width_shape))
    frame_XY = np.stack([X.T, Y], axis=-1)
    return frame_XY


def _projXY2panoXYZ(projX: float, projY: float, proj_params: proto_gen.detect_pb2.StereoProjectParams) \
        -> (float, float, float):
    """
    视点坐标：（1，0，0）；投影平面：x = -proj_dis
    (project_size / 2, project_size / 2) 时 proj_line = 0, 会出现问题
    :param projX: [0, proj_size]，从左到右
    :param projY: [0, proj_size]，从上到下
    :return:
    """
    # 投影平面上的点坐标
    proj_point = (
        -proj_params.project_dis,
        proj_params.project_size / 2 - projX,
        proj_params.project_size / 2 - projY,
    )
    # 投影点与投影中心的距离的平方
    proj_line = math.sqrt(
        math.pow(proj_point[1], 2) +
        math.pow(proj_point[2], 2)
    )
    if proj_line == 0:
        return -proj_params.project_dis, 0, 0
    # 当前投影光线与中心线的夹角 theta
    tan_angle_ray_center = proj_line / (1 + proj_params.project_dis)
    # 中间变量 tan2
    tan2 = math.pow(tan_angle_ray_center, 2)
    # 计算 x
    panoX = -(1 - tan2) / (1 + tan2)  # r * cos2theta
    # 中间变量 r*sin2theta
    r_sin2theta = 2 * tan_angle_ray_center / (1 + tan2)  # r * sin2theta
    # 计算 y, z
    cos_line_xy = proj_point[1] / proj_line  # 投影点与投影中心连线与 xy 平面的夹角 phi
    panoY = r_sin2theta * cos_line_xy
    sin_line_xy = proj_point[2] / proj_line  # 投影点与投影中心连线与 xy 平面的夹角 phi
    panoZ = r_sin2theta * sin_line_xy

    return (panoX, panoY, panoZ)


def _projXY2panoXYZ_nda(projXY: np.ndarray, proj_params: proto_gen.detect_pb2.StereoProjectParams) -> np.ndarray:
    """
    将投影坐标转换为全景坐标
    Args:
        projXY: 投影坐标，[X, Y]
        proj_params: 投影参数

    Returns: 全景坐标，[X, Y, Z]

    """
    # 向量拆分
    projX, projY = np.split(projXY, 2, axis=-1)
    # 投影平面上的点坐标
    proj_point_Y = - projX + proj_params.project_size / 2
    proj_point_Z = - projY + proj_params.project_size / 2
    # 投影点与投影中心的距离的平方
    proj_line = np.sqrt(
        np.power(proj_point_Y, 2) +
        np.power(proj_point_Z, 2)
    )
    # 当前投影光线与中心线的夹角 theta
    tan_angle_ray_center = proj_line / (1 + proj_params.project_dis)
    # 中间变量 tan2
    tan2 = np.power(tan_angle_ray_center, 2)
    # 计算 x
    panoX = -(1 - tan2) / (1 + tan2)  # r * cos2theta
    # 中间变量 r*sin2theta
    r_sin2theta = 2 * tan_angle_ray_center / (1 + tan2)  # r * sin2theta
    # 计算 y, z
    cos_line_xy = proj_point_Y / proj_line  # 投影点与投影中心连线与 xy 平面的夹角 phi
    panoY = r_sin2theta * cos_line_xy
    sin_line_xy = proj_point_Z / proj_line  # 投影点与投影中心连线与 xy 平面的夹角 phi
    panoZ = r_sin2theta * sin_line_xy
    # 向量聚合
    return np.concatenate([panoX, panoY, panoZ], axis=-1)


def _map_pano_XY(pano_image: np.ndarray, proj_frame_pano_XY: np.ndarray) -> np.ndarray:
    """
    将全景图像映射到投影图像
    xyz:                [-1,1], [-1,1], [-1,1], 单位球面, 右手坐标系, z 轴向上
    u, v:               [0,1], [0,1), u 向下, v 向右
    phi, theta:         [0,pi], [0, 2pi), phi 与 z 轴夹角, theta 与 x 轴夹角
    Args:
        pano_image:     全景图像
        proj_frame_XY:  投影图像的全景坐标，[X, Y]

    Returns: 投影图像

    """
    X, Y = np.split(proj_frame_pano_XY, 2, axis=-1)
    X = np.squeeze(X)
    Y = np.squeeze(Y)

    # 因为全景图是首位相连的，所以这里 mode='wrap'
    mc0 = ndimage.map_coordinates(pano_image[:, :, 0], [Y, X], mode='wrap')  # channel: B
    mc1 = ndimage.map_coordinates(pano_image[:, :, 1], [Y, X], mode='wrap')  # channel: G
    mc2 = ndimage.map_coordinates(pano_image[:, :, 2], [Y, X], mode='wrap')  # channel: R

    output = np.stack([mc0, mc1, mc2], axis=-1)
    return output


@proj.proj_utils.proj_count_and_time
def stereo_proj(im: np.ndarray,
                proj_params: proto_gen.detect_pb2.StereoProjectParams,
                proj_width: int, proj_height: int) -> np.ndarray:
    """
    Project an image into a stereo image.
    Args:
        im: The image to project.
        proj_params: The stereo projection parameters.
        proj_width: The width of the projection image.
        proj_height: The height of the projection image.

    Returns: The projected image.
    """
    im_height, im_width = im.shape[:2]
    # 投影参数
    proj_params_key = proj_params.SerializeToString()
    # 查找投影缓存
    if proj_params_key in StereoProjCache:
        logging.debug(f"cache in {proj_params_key}")
        # 投影帧采样全景 XY
        proj_frame_pano_XY = pickle.loads(StereoProjCache[proj_params_key])
    else:
        logging.debug(f"cache miss {proj_params_key}")
        # 投影帧采样投影 XY
        proj_frame_proj_XY = _gen_frame_XY(
            proj_params.project_size, proj_params.project_size,
            proj_width, proj_height)
        # 投影帧采样全景 xyz
        proj_frame_pano_xyz = _projXY2panoXYZ_nda(proj_frame_proj_XY, proj_params)
        # 投影帧采样全景 XY
        proj_frame_pano_XY = proj.transform.xyz2XY_nda(
            proj_frame_pano_xyz, width=im_width, height=im_height,
            theta_rotate=proj_params.theta_rotate)
        StereoProjCache[proj_params_key] = pickle.dumps(proj_frame_pano_XY)

    project_image_ndarray = _map_pano_XY(im, proj_frame_pano_XY)

    return project_image_ndarray


def _panoXYZ2projXY(panoX: float, panoY: float, panoZ: float, proj_params: proto_gen.detect_pb2.StereoProjectParams) \
        -> (float, float, bool):
    """
    不考虑 theta_rotate
    Args:
        panoX:
        panoY:
        panoZ:
        proj_params:

    Returns:
        [0, proj_size]

    """
    proj_point_x = -proj_params.project_dis
    proj_point_scale = (1 - proj_point_x) / (1 - panoX)
    proj_point_y = panoY * proj_point_scale
    proj_point_z = panoZ * proj_point_scale
    # [proj_params.project_size / 2, - proj_params.project_size / 2] -> [0, proj_params.project_size]
    projX = proj_params.project_size / 2 - proj_point_y
    projY = proj_params.project_size / 2 - proj_point_z
    # 是否在投影框内
    inProj = 0 <= projX <= proj_params.project_size and 0 <= projY <= proj_params.project_size
    return projX, projY, inProj


########## Test2 ##########
def func2():
    # pano_height * pano_width * 3
    input_pano = cv2.imread(
        # "/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/proj/demo/mask/pano_0a46e210e7018af58de6f45f0997486c_1_9_sofa.png")
        "/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/proj/demo/image/pano_0a46e210e7018af58de6f45f0997486c.png")
    # "/Users/limengfan/PycharmProjects/PanoDetectionProtobuf/proj/demo/image/pano_0a46e210e7018af58de6f45f0997486c.png")

    proj_params_list = []
    total_theta_step = 24
    for project_dis in range(1, 2):
        for project_size in range(2, 3):
            for theta_step in range(total_theta_step):
                proj_params_list.extend(
                    [
                        proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
                                                                 theta_rotate=theta_step * np.pi * 2 / total_theta_step),
                    ]
                )

    import time
    import utils.plot

    output_proj_stack = []
    for i, proj_params in enumerate(proj_params_list):
        curTime = time.time()
        output_proj = stereo_proj(input_pano,
                                  proj_params=proj_params,
                                  proj_width=640, proj_height=640)
        logging.info(f'time cost {i} {time.time() - curTime} s')
        output_proj_stack.append(output_proj)
    cv2.imshow(f"input_pano_stereo", input_pano)
    cv2.imshow(f"output_proj_stack_stereo", utils.plot.plot_nda_list(output_proj_stack, width_size=total_theta_step))
    cv2.waitKey(0)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.ERROR,
    )

    func2()
