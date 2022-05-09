"""
xyz:                [-1,1], [-1,1], [-1,1], 单位球面, 右手坐标系, z 轴向上
phi, theta:         [0,pi], [0,2pi), phi 与 z 轴夹角, theta 与 x 轴夹角, theta_rotate 为顺时针旋转的角度
X, Y:               [0,width), [0,height], X 向右, Y 向下, width 不能取到，否则就会出现 theta 的 0 和 2pi 重合
"""

import numpy as np


def theta_phi2xyz(theta: float, phi: float, theta_rotate: float = 0) \
        -> (float, float, float):
    # 全景图水平顺时针旋转
    if theta >= theta_rotate:
        theta -= theta_rotate
    else:
        theta = 2 * np.pi - theta_rotate
    # 坐标映射
    z = np.cos(phi)
    rP = np.sin(phi)  # 投影半径
    x = np.cos(theta) * rP
    y = np.sin(theta) * rP
    return x, y, z


def theta_phi2xyz_nda(theta_phi: np.ndarray, theta_rotate: float = 0) \
        -> np.ndarray:
    # 向量拆分
    theta, phi = np.split(theta_phi, 2, axis=-1)
    # 全景图水平顺时针旋转
    theta = np.vectorize(
        lambda theta: (theta - theta_rotate) if theta >= theta_rotate else 2 * np.pi - theta_rotate
    )(theta)
    # 坐标映射
    z = np.cos(phi)
    rP = np.sin(phi)  # 投影半径
    x = np.cos(theta) * rP
    y = np.sin(theta) * rP
    # 向量聚合
    return np.concatenate([x, y, z], axis=-1)


def xyz2theta_phi(x: float, y: float, z: float, theta_rotate: float = 0) \
        -> (float, float):
    # 坐标映射
    theta = np.arctan2(y, x)
    phi = np.arccos(z)
    # [-pi, pi) -> [0, 2pi)；x 轴左侧 [0, -pi] -> [2pi, pi]；x 轴右侧不变
    theta = (2 * np.pi + theta) if theta < 0 else theta
    # 全景图水平逆时针旋转
    theta += theta_rotate
    if theta >= 2 * np.pi:
        theta -= 2 * np.pi
    return theta, phi


def xyz2theta_phi_nda(xyz: np.ndarray, theta_rotate: float = 0) -> np.ndarray:
    # 向量拆分
    x, y, z = np.split(xyz, 3, axis=-1)
    # 坐标映射
    phi = np.arccos(z)
    theta = np.arctan2(y, x)
    # [-pi, pi) -> [0, 2pi)；x 轴左侧 [0, -pi] -> [2pi, pi]；x 轴右侧不变
    theta = np.vectorize(
        lambda theta: (2 * np.pi + theta) if theta < 0 else theta
    )(theta)
    # 全景图水平逆时针旋转
    theta += theta_rotate
    theta = np.vectorize(
        lambda theta: (theta - 2 * np.pi) if theta >= 2 * np.pi else theta
    )(theta)
    a = theta.T
    # 向量聚合
    return np.concatenate([theta, phi], axis=-1)


def theta_phi2XY(theta: float, phi: float, width: float = 1, height: float = 1) \
        -> (float, float):
    # 坐标映射
    X = theta / (2 * np.pi) * width
    Y = phi / np.pi * height
    return X, Y


def theta_phi2XY_nda(theta_phi: np.ndarray, width: float = 1, height: float = 1) -> np.ndarray:
    # 向量拆分
    theta, phi = np.split(theta_phi, 2, axis=-1)
    # 坐标映射
    X = theta / (2 * np.pi) * width
    Y = phi / np.pi * height
    # 向量聚合
    return np.concatenate([X, Y], axis=-1)


def XY2theta_phi(X: float, Y: float, width: float = 1, height: float = 1) \
        -> (float, float):
    # 坐标映射
    theta = X / width * 2 * np.pi
    phi = Y / height * np.pi
    return theta, phi


def XY2theta_phi_nda(XY: np.ndarray, width: float = 1, height: float = 1) -> np.ndarray:
    # 向量拆分
    X, Y = np.split(XY, 2, axis=-1)
    # 坐标映射
    theta = X / width * 2 * np.pi
    phi = Y / height * np.pi
    # 向量聚合
    return np.concatenate([theta, phi], axis=-1)


def xyz2XY(x: float, y: float, z: float, width: float = 1, height: float = 1, theta_rotate: float = 0) \
        -> (float, float):
    # 坐标映射
    theta, phi = xyz2theta_phi(x, y, z, theta_rotate=theta_rotate)
    X, Y = theta_phi2XY(theta, phi, width=width, height=height)
    return X, Y


def xyz2XY_nda(xyz: np.ndarray, width: float = 1, height: float = 1, theta_rotate: float = 0) -> np.ndarray:
    # 坐标映射
    theta_phi = xyz2theta_phi_nda(xyz, theta_rotate=theta_rotate)
    XY = theta_phi2XY_nda(theta_phi, width=width, height=height)
    return XY


def XY2xyz(X: float, Y: float, width: float = 1, height: float = 1, theta_rotate: float = 0) \
        -> (float, float, float):
    # 坐标映射
    theta, phi = XY2theta_phi(X, Y, width=width, height=height)
    x, y, z = theta_phi2xyz(theta, phi, theta_rotate=theta_rotate)
    return x, y, z


def XY2xyz_nda(XY: np.ndarray, width: float = 1, height: float = 1, theta_rotate: float = 0) -> np.ndarray:
    # 坐标映射
    theta_phi = XY2theta_phi_nda(XY, width=width, height=height)
    xyz = theta_phi2xyz_nda(theta_phi, theta_rotate=theta_rotate)
    return xyz


########## test ##########
if __name__ == "__main__":
    print(XY2xyz(378, 464, 1024, 512))
    exit(1)

    import numpy as np

    for Y in np.arange(0, 1, 0.01):
        for X in np.arange(0, 1, 0.01):
            print(f"XY: (%.2f, %.2f), theta_phi: (%.2f, %.2f), xyz: (%.2f, %.2f, %.2f)" % (
                X, Y, *XY2theta_phi(X, Y), *XY2xyz(X, Y)
            ))
