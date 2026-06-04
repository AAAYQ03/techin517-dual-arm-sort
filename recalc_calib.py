import numpy as np
from scipy.spatial.transform import Rotation as R

def make_T(t, q):
    T = np.eye(4)
    T[:3,:3] = R.from_quat(q).as_matrix()
    T[:3,3] = t
    return T

# 标定节点输出: optical -> base_link (ROS 约定: base_link 在 optical 系下)
T_optical_base_RAW = make_T(
    [0.0204, -0.1066, 0.3383],
    [-0.6185, 0.6737, -0.2659, 0.3049]
)

# 求逆: optical 在 base 系下
T_base_optical = np.linalg.inv(T_optical_base_RAW)

# 相机内部 TF: optical 在 cam_link 系下
T_camlink_optical = make_T(
    [0, 0.015, 0],
    [-0.497, 0.504, -0.497, 0.502]
)

# cam_link 在 base 系下 (launch 需要的)
T_base_camlink = T_base_optical @ np.linalg.inv(T_camlink_optical)

t = T_base_camlink[:3, 3]
q = R.from_matrix(T_base_camlink[:3,:3]).as_quat()
print("==== NEW (with inverse) ====")
print(f"translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
print(f"quat (xyzw): [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
print()
print("==== OLD (current launch values) ====")
print("translation: [-0.0780, 0.2623, 0.2256]")
print("quat (xyzw): [-0.2612, 0.6671, -0.3081, 0.6260]")
