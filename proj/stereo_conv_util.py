import logging
from typing import List

import cv2
import numpy as np

import proto_gen.detect_pb2
import proj.stereo_proj
import proj.transform


# 获取投影参数对应的全景图像 theta 和 phi 的范围，不考虑 theta_rotate
def get_theta_phi_range(proj_params: proto_gen.detect_pb2.StereoProjectParams) -> (float, float, float, float):
    """
    Get theta and phi range for the given stereo projection parameters.
    Args:
        proj_params: Stereo projection parameters.

    Returns:
        Theta and phi begin.
        Theta and phi range.
    """
    proj_params.theta_rotate = 0  # 不考虑 theta_rotate
    # 判断球顶
    _, _, inProj = proj.stereo_proj._panoXYZ2projXY(0, 0, 1, proj_params)
    if inProj:
        logging.warning('bad proj when the (0,0,1) is in the projection plane')
        return 0, 0, 2 * np.pi, np.pi
    # theta
    xmin, ymin, zmin = proj.stereo_proj._projXY2panoXYZ(0, 0, proj_params)
    xmax, ymax, zmax = proj.stereo_proj._projXY2panoXYZ(proj_params.project_size, proj_params.project_size, proj_params)
    theta_min, phi_min = proj.transform.xyz2theta_phi(xmin, ymin, zmin)
    theta_max, phi_max = proj.transform.xyz2theta_phi(xmax, ymax, zmax)
    # phi
    xmin, ymin, zmin = proj.stereo_proj._projXY2panoXYZ(proj_params.project_size / 2, 0, proj_params)
    xmax, ymax, zmax = proj.stereo_proj._projXY2panoXYZ(proj_params.project_size / 2, proj_params.project_size,
                                                        proj_params)
    _, phi_min = proj.transform.xyz2theta_phi(xmin, ymin, zmin)
    _, phi_max = proj.transform.xyz2theta_phi(xmax, ymax, zmax)

    return theta_min, phi_min, theta_max - theta_min, phi_max - phi_min


def plot_delta_theta_phi(proj_req: proto_gen.detect_pb2.StereoProjectRequest, delta_size=100) -> List[np.ndarray]:
    im_raw = cv2.imread(proj_req.project_request.pano_dataset_model.image_path)  # BGR

    ret = []

    for proj_params in proj_req.project_params_list:
        im = proj.stereo_proj.stereo_proj(im_raw, proj_params,
                                          proj_req.project_request.project_width,
                                          proj_req.project_request.project_height)
        theta_min, phi_min, theta_range, phi_range = get_theta_phi_range(proj_params)
        for delta_theta in np.linspace(0, theta_range, delta_size):
            for delta_phi in np.linspace(0, phi_range, delta_size):
                theta = theta_min + delta_theta
                phi = phi_min + delta_phi
                # for delta_theta in np.linspace(0, 2 * np.pi, delta_size):
                #     for delta_phi in np.linspace(0, np.pi, delta_size):
                #         heta = delta_theta
                #         phi = delta_phi
                x, y, z = proj.transform.theta_phi2xyz(theta, phi, theta_rotate=proj_params.theta_rotate)
                X, Y, _ = proj.stereo_proj._panoXYZ2projXY(x, y, z, proj_params)
                projX = int(X / proj_params.project_size * proj_req.project_request.project_width)
                projY = int(Y / proj_params.project_size * proj_req.project_request.project_height)
                cv2.circle(im, (projX, projY), 1, (0, 0, 255), -1)
        ret.append(im)
    return ret


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO,
    )

    proj_params_list = []
    total_theta_step = 24
    for project_dis in range(1, 2):
        for project_size in range(1, 4):
            for theta_step in range(total_theta_step):
                proj_params_list.extend(
                    [
                        proto_gen.detect_pb2.StereoProjectParams(project_dis=project_dis, project_size=project_size,
                                                                 theta_rotate=theta_step * np.pi * 2 / total_theta_step),
                    ]
                )

    proj_list = plot_delta_theta_phi(
        proto_gen.detect_pb2.StereoProjectRequest(
            project_request=proto_gen.detect_pb2.ProjectRequest(
                pano_dataset_model=proto_gen.detect_pb2.DatasetModel(
                    image_path="/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/proj/demo/image/pano_0a46e210e7018af58de6f45f0997486c.png",
                ),
                pano_height=512,
                pano_width=1024,
                project_height=640,
                project_width=640,
            ),
            project_params_list=proj_params_list,
        )
    )

    import utils.plot

    plot_im=utils.plot.plot_nda_list(proj_list, width_size=total_theta_step)

    cv2.imshow("win",plot_im )
    cv2.imwrite("/Users/bytedance/Desktop/delta_theta_phi_2.png",plot_im)
    cv2.waitKey(0)
