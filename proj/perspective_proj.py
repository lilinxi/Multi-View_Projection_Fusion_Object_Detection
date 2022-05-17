import logging

import pickle

import cv2
import numpy as np

import proto_gen.detect_pb2

import proj.transform
import proj.stereo_proj
import proj.proj_utils

PerspectiveProjCache = {}


def _projXY2panoXYZ(projX: float, projY: float, proj_params: proto_gen.detect_pb2.StereoProjectParams) \
        -> (float, float, float):
    """
    将投影坐标转换为全景坐标
    视点坐标：（1，0，0）；投影平面：x = -proj_dis
    Args:
        projX: [0, proj_size]，从左到右
        projY: [0, proj_size]，从上到下
        proj_params:

    Returns: 全景坐标，[X, Y, Z]
    """
    # 投影平面上的点坐标
    proj_point = (
        -proj_params.project_dis,
        proj_params.project_size / 2 - projX,
        proj_params.project_size / 2 - projY,
    )
    # 投影点与球心的距离的平方
    point_line = np.sqrt(
        np.power(proj_point[0], 2) +
        np.power(proj_point[1], 2) +
        np.power(proj_point[2], 2)
    )
    # 计算全景球面 xyz
    panoX = proj_point[0] / point_line
    panoY = proj_point[1] / point_line
    panoZ = proj_point[2] / point_line
    return (panoX, panoY, panoZ)


def _projXY2panoXYZ_nda(projXY: np.ndarray, proj_params: proto_gen.detect_pb2.StereoProjectParams) -> np.ndarray:
    """
    将投影坐标转换为全景坐标
    视点坐标：（1，0，0）；投影平面：x = -proj_dis
    Args:
        projX: [0, proj_size]，从左到右
        projY: [0, proj_size]，从上到下
        proj_params:

    Returns: 全景坐标，[X, Y, Z]
    """
    # 向量拆分
    projX, projY = np.split(projXY, 2, axis=-1)
    # 投影平面上的点坐标
    proj_point_X = np.ones(projX.shape, dtype=projX.dtype) * - proj_params.project_dis
    proj_point_Y = - projX + proj_params.project_size / 2
    proj_point_Z = - projY + proj_params.project_size / 2
    # 投影点与投影中心的距离的平方
    point_line = np.sqrt(
        np.power(proj_point_X, 2) +
        np.power(proj_point_Y, 2) +
        np.power(proj_point_Z, 2)
    )
    # 计算全景球面 xyz
    panoX = proj_point_X / point_line
    panoY = proj_point_Y / point_line
    panoZ = proj_point_Z / point_line
    # 向量聚合
    return np.concatenate([panoX, panoY, panoZ], axis=-1)


@proj.proj_utils.proj_count_and_time
def perspective_proj(im: np.ndarray,
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
    if proj_params_key in PerspectiveProjCache:
        logging.debug(f"cache in {proj_params_key}")
        # 投影帧采样全景 XY
        proj_frame_pano_XY = pickle.loads(PerspectiveProjCache[proj_params_key])
    else:
        logging.debug(f"cache miss {proj_params_key}")
        # 投影帧采样投影 XY
        proj_frame_proj_XY = proj.stereo_proj._gen_frame_XY(
            proj_params.project_size, proj_params.project_size,
            proj_width, proj_height)
        # 投影帧采样全景 xyz
        proj_frame_pano_xyz = _projXY2panoXYZ_nda(proj_frame_proj_XY, proj_params)
        # 投影帧采样全景 XY
        proj_frame_pano_XY = proj.transform.xyz2XY_nda(
            proj_frame_pano_xyz, width=im_width, height=im_height,
            theta_rotate=proj_params.theta_rotate)
        PerspectiveProjCache[proj_params_key] = pickle.dumps(proj_frame_pano_XY)

    project_image_ndarray = proj.stereo_proj._map_pano_XY(im, proj_frame_pano_XY)

    return project_image_ndarray


########## Test2 ##########
def func2():
    # pano_height * pano_width * 3
    input_pano = cv2.imread(
        # "/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/proj/demo/mask/pano_0a46e210e7018af58de6f45f0997486c_1_9_sofa.png")
        "/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/proj/demo/image/pano_0a46e210e7018af58de6f45f0997486c.png")

    import time

    curTime = time.time()

    for i, theta_rotate in enumerate(range(8)):
        theta_rotate = np.pi * theta_rotate / 8
        output_proj = perspective_proj(input_pano,
                                       proto_gen.detect_pb2.StereoProjectParams(
                                           project_dis=1, project_size=4, theta_rotate=theta_rotate),
                                       proj_width=800, proj_height=600)

        cv2.imshow(f"output_proj{i}", output_proj)
    logging.info(f'time cost {time.time() - curTime} s')
    cv2.imshow(f"input_pano", input_pano)
    cv2.waitKey(0)


########## Test3 ##########
def func3():
    # pano_height * pano_width * 3
    input_pano = cv2.imread(
        # "/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/proj/demo/mask/pano_0a46e210e7018af58de6f45f0997486c_1_9_sofa.png")
        "/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/proj/demo/image/pano_0a46e210e7018af58de6f45f0997486c.png")
    # "/Users/limengfan/PycharmProjects/PanoDetectionProtobuf/proj/demo/image/pano_0a46e210e7018af58de6f45f0997486c.png")

    # proj_params_list = []
    # for project_dis in range(1, 2):
    #     for project_size in range(project_dis, project_dis + 6):
    #         proj_params_list.extend(
    #             [
    #                 proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
    #                                                          theta_rotate=0.0),
    #                 proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
    #                                                          theta_rotate=np.pi / 4),
    #                 proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
    #                                                          theta_rotate=np.pi / 2),
    #                 proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
    #                                                          theta_rotate=np.pi * 3 / 4),
    #                 proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
    #                                                          theta_rotate=np.pi),
    #                 proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
    #                                                          theta_rotate=np.pi * 5 / 4),
    #                 proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
    #                                                          theta_rotate=np.pi * 3 / 2),
    #                 proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
    #                                                          theta_rotate=np.pi * 7 / 4)
    #             ]
    #         )
    proj_params_list = []
    total_theta_step = 24
    for project_dis in range(1, 2):
        for project_size in range(project_dis, project_dis + 6):
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
        output_proj = perspective_proj(input_pano,
                                       proj_params=proj_params,
                                       proj_width=640, proj_height=640)
        logging.info(f'time cost {i} {time.time() - curTime} s')
        output_proj_stack.append(output_proj)
    cv2.imshow(f"input_pano", input_pano)
    cv2.imshow(f"output_proj_stack", utils.plot.plot_nda_list(output_proj_stack, width_size=total_theta_step))
    cv2.imwrite(f"/Users/bytedance/Desktop/output_proj_stack.png",
                utils.plot.plot_nda_list(output_proj_stack, width_size=total_theta_step))
    cv2.waitKey(0)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.ERROR,
    )

    func3()
    func3()

    """
    cacha in 0.16
    cacha miss 0.45
    /usr/local/bin/python3 /Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/proj/perspective_proj.py
25-03-2022:12:23:11,672 INFO     [perspective_proj.py:182] time cost 0 0.5087459087371826 s
25-03-2022:12:23:12,78 INFO     [perspective_proj.py:182] time cost 1 0.40518808364868164 s
25-03-2022:12:23:12,473 INFO     [perspective_proj.py:182] time cost 2 0.39543890953063965 s
25-03-2022:12:23:12,876 INFO     [perspective_proj.py:182] time cost 3 0.4026679992675781 s
25-03-2022:12:23:13,278 INFO     [perspective_proj.py:182] time cost 4 0.4021580219268799 s
25-03-2022:12:23:13,700 INFO     [perspective_proj.py:182] time cost 5 0.4218270778656006 s
25-03-2022:12:23:14,117 INFO     [perspective_proj.py:182] time cost 6 0.4168992042541504 s
25-03-2022:12:23:14,538 INFO     [perspective_proj.py:182] time cost 7 0.4203002452850342 s
25-03-2022:12:23:14,916 INFO     [perspective_proj.py:182] time cost 8 0.3785979747772217 s
25-03-2022:12:23:15,296 INFO     [perspective_proj.py:182] time cost 9 0.379680871963501 s
25-03-2022:12:23:15,746 INFO     [perspective_proj.py:182] time cost 10 0.4493551254272461 s
25-03-2022:12:23:16,231 INFO     [perspective_proj.py:182] time cost 11 0.48557496070861816 s
25-03-2022:12:23:16,636 INFO     [perspective_proj.py:182] time cost 12 0.4045441150665283 s
25-03-2022:12:23:17,70 INFO     [perspective_proj.py:182] time cost 13 0.43398404121398926 s
25-03-2022:12:23:17,494 INFO     [perspective_proj.py:182] time cost 14 0.4236931800842285 s
25-03-2022:12:23:17,907 INFO     [perspective_proj.py:182] time cost 15 0.41304802894592285 s
25-03-2022:12:23:18,282 INFO     [perspective_proj.py:182] time cost 16 0.3748910427093506 s
25-03-2022:12:23:18,657 INFO     [perspective_proj.py:182] time cost 17 0.37471795082092285 s
25-03-2022:12:23:19,31 INFO     [perspective_proj.py:182] time cost 18 0.3742351531982422 s
25-03-2022:12:23:19,415 INFO     [perspective_proj.py:182] time cost 19 0.38331174850463867 s
25-03-2022:12:23:19,816 INFO     [perspective_proj.py:182] time cost 20 0.4012300968170166 s
25-03-2022:12:23:20,227 INFO     [perspective_proj.py:182] time cost 21 0.4111289978027344 s
25-03-2022:12:23:20,643 INFO     [perspective_proj.py:182] time cost 22 0.416245698928833 s
25-03-2022:12:23:21,63 INFO     [perspective_proj.py:182] time cost 23 0.41906094551086426 s
25-03-2022:12:23:21,447 INFO     [perspective_proj.py:182] time cost 24 0.3846449851989746 s
25-03-2022:12:23:21,824 INFO     [perspective_proj.py:182] time cost 25 0.3768928050994873 s
25-03-2022:12:23:22,206 INFO     [perspective_proj.py:182] time cost 26 0.3812088966369629 s
25-03-2022:12:23:22,592 INFO     [perspective_proj.py:182] time cost 27 0.38597607612609863 s
25-03-2022:12:23:22,989 INFO     [perspective_proj.py:182] time cost 28 0.39748573303222656 s
25-03-2022:12:23:23,396 INFO     [perspective_proj.py:182] time cost 29 0.40651798248291016 s
25-03-2022:12:23:23,813 INFO     [perspective_proj.py:182] time cost 30 0.4169750213623047 s
25-03-2022:12:23:24,233 INFO     [perspective_proj.py:182] time cost 31 0.4193131923675537 s
25-03-2022:12:23:24,613 INFO     [perspective_proj.py:182] time cost 32 0.3808310031890869 s
25-03-2022:12:23:24,991 INFO     [perspective_proj.py:182] time cost 33 0.37782812118530273 s
25-03-2022:12:23:25,368 INFO     [perspective_proj.py:182] time cost 34 0.3764479160308838 s
25-03-2022:12:23:25,779 INFO     [perspective_proj.py:182] time cost 35 0.4108548164367676 s
25-03-2022:12:23:26,178 INFO     [perspective_proj.py:182] time cost 36 0.39856910705566406 s
25-03-2022:12:23:26,580 INFO     [perspective_proj.py:182] time cost 37 0.40276312828063965 s
25-03-2022:12:23:26,997 INFO     [perspective_proj.py:182] time cost 38 0.4168682098388672 s
25-03-2022:12:23:27,419 INFO     [perspective_proj.py:182] time cost 39 0.4211740493774414 s
25-03-2022:12:23:27,796 INFO     [perspective_proj.py:182] time cost 40 0.37686777114868164 s
25-03-2022:12:23:28,172 INFO     [perspective_proj.py:182] time cost 41 0.37642860412597656 s
25-03-2022:12:23:28,547 INFO     [perspective_proj.py:182] time cost 42 0.37509822845458984 s
25-03-2022:12:23:28,938 INFO     [perspective_proj.py:182] time cost 43 0.3909478187561035 s
25-03-2022:12:23:29,337 INFO     [perspective_proj.py:182] time cost 44 0.39893507957458496 s
25-03-2022:12:23:29,746 INFO     [perspective_proj.py:182] time cost 45 0.408627986907959 s
25-03-2022:12:23:30,162 INFO     [perspective_proj.py:182] time cost 46 0.41521787643432617 s
25-03-2022:12:23:30,577 INFO     [perspective_proj.py:182] time cost 47 0.4156641960144043 s
25-03-2022:12:23:34,224 INFO     [perspective_proj.py:182] time cost 0 0.17647409439086914 s
25-03-2022:12:23:34,394 INFO     [perspective_proj.py:182] time cost 1 0.1701500415802002 s
25-03-2022:12:23:34,566 INFO     [perspective_proj.py:182] time cost 2 0.17185211181640625 s
25-03-2022:12:23:34,729 INFO     [perspective_proj.py:182] time cost 3 0.16347002983093262 s
25-03-2022:12:23:34,897 INFO     [perspective_proj.py:182] time cost 4 0.16730999946594238 s
25-03-2022:12:23:35,60 INFO     [perspective_proj.py:182] time cost 5 0.16324806213378906 s
25-03-2022:12:23:35,226 INFO     [perspective_proj.py:182] time cost 6 0.1660299301147461 s
25-03-2022:12:23:35,386 INFO     [perspective_proj.py:182] time cost 7 0.15973901748657227 s
25-03-2022:12:23:35,550 INFO     [perspective_proj.py:182] time cost 8 0.16433405876159668 s
25-03-2022:12:23:35,712 INFO     [perspective_proj.py:182] time cost 9 0.16162776947021484 s
25-03-2022:12:23:35,878 INFO     [perspective_proj.py:182] time cost 10 0.16559219360351562 s
25-03-2022:12:23:36,40 INFO     [perspective_proj.py:182] time cost 11 0.1616499423980713 s
25-03-2022:12:23:36,210 INFO     [perspective_proj.py:182] time cost 12 0.17066526412963867 s
25-03-2022:12:23:36,373 INFO     [perspective_proj.py:182] time cost 13 0.16222000122070312 s
25-03-2022:12:23:36,536 INFO     [perspective_proj.py:182] time cost 14 0.16329598426818848 s
25-03-2022:12:23:36,697 INFO     [perspective_proj.py:182] time cost 15 0.1610121726989746 s
25-03-2022:12:23:36,860 INFO     [perspective_proj.py:182] time cost 16 0.16250181198120117 s
25-03-2022:12:23:37,22 INFO     [perspective_proj.py:182] time cost 17 0.1619701385498047 s
25-03-2022:12:23:37,184 INFO     [perspective_proj.py:182] time cost 18 0.1625070571899414 s
25-03-2022:12:23:37,360 INFO     [perspective_proj.py:182] time cost 19 0.17513298988342285 s
25-03-2022:12:23:37,522 INFO     [perspective_proj.py:182] time cost 20 0.16269707679748535 s
25-03-2022:12:23:37,682 INFO     [perspective_proj.py:182] time cost 21 0.15942621231079102 s
25-03-2022:12:23:37,845 INFO     [perspective_proj.py:182] time cost 22 0.16335606575012207 s
25-03-2022:12:23:38,9 INFO     [perspective_proj.py:182] time cost 23 0.16353201866149902 s
25-03-2022:12:23:38,177 INFO     [perspective_proj.py:182] time cost 24 0.16819524765014648 s
25-03-2022:12:23:38,338 INFO     [perspective_proj.py:182] time cost 25 0.16061639785766602 s
25-03-2022:12:23:38,502 INFO     [perspective_proj.py:182] time cost 26 0.16373395919799805 s
25-03-2022:12:23:38,663 INFO     [perspective_proj.py:182] time cost 27 0.16108417510986328 s
25-03-2022:12:23:38,829 INFO     [perspective_proj.py:182] time cost 28 0.16576313972473145 s
25-03-2022:12:23:38,996 INFO     [perspective_proj.py:182] time cost 29 0.16688299179077148 s
25-03-2022:12:23:39,161 INFO     [perspective_proj.py:182] time cost 30 0.16487383842468262 s
25-03-2022:12:23:39,322 INFO     [perspective_proj.py:182] time cost 31 0.1610090732574463 s
25-03-2022:12:23:39,495 INFO     [perspective_proj.py:182] time cost 32 0.172943115234375 s
25-03-2022:12:23:39,658 INFO     [perspective_proj.py:182] time cost 33 0.16325902938842773 s
25-03-2022:12:23:39,822 INFO     [perspective_proj.py:182] time cost 34 0.16394495964050293 s
25-03-2022:12:23:39,986 INFO     [perspective_proj.py:182] time cost 35 0.16409587860107422 s
25-03-2022:12:23:40,151 INFO     [perspective_proj.py:182] time cost 36 0.1641678810119629 s
25-03-2022:12:23:40,313 INFO     [perspective_proj.py:182] time cost 37 0.16161799430847168 s
25-03-2022:12:23:40,476 INFO     [perspective_proj.py:182] time cost 38 0.16323208808898926 s
25-03-2022:12:23:40,637 INFO     [perspective_proj.py:182] time cost 39 0.16103792190551758 s
25-03-2022:12:23:40,805 INFO     [perspective_proj.py:182] time cost 40 0.16791224479675293 s
25-03-2022:12:23:40,969 INFO     [perspective_proj.py:182] time cost 41 0.16422605514526367 s
25-03-2022:12:23:41,137 INFO     [perspective_proj.py:182] time cost 42 0.16810393333435059 s
25-03-2022:12:23:41,302 INFO     [perspective_proj.py:182] time cost 43 0.1640620231628418 s
25-03-2022:12:23:41,466 INFO     [perspective_proj.py:182] time cost 44 0.16443371772766113 s
25-03-2022:12:23:41,629 INFO     [perspective_proj.py:182] time cost 45 0.16233611106872559 s
25-03-2022:12:23:41,794 INFO     [perspective_proj.py:182] time cost 46 0.16494512557983398 s
25-03-2022:12:23:41,957 INFO     [perspective_proj.py:182] time cost 47 0.16363024711608887 s

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
    """
