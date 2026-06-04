import numpy as np
from scipy.spatial.transform import Rotation as R

def make_T(t, q):
    T = np.eye(4)
    T[:3,:3] = R.from_quat(q).as_matrix()
    T[:3,3] = t
    return T

raw_t = [0.0204, -0.1066, 0.3383]
raw_q = [-0.6185, 0.6737, -0.2659, 0.3049]
T_RAW = make_T(raw_t, raw_q)

T_camlink_optical = make_T(
    [0, 0.015, 0],
    [-0.497, 0.504, -0.497, 0.502]
)

# 4 种可能的方向组合
print("=== Variant 1: (RAW) @ inv(camlink_optical) — 当时 PDF 的版本 ===")
T1 = T_RAW @ np.linalg.inv(T_camlink_optical)
t = T1[:3, 3]; q = R.from_matrix(T1[:3,:3]).as_quat()
print(f"  t: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]   q: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")

print("=== Variant 2: inv(RAW) @ inv(camlink_optical) — 求逆版本 ===")
T2 = np.linalg.inv(T_RAW) @ np.linalg.inv(T_camlink_optical)
t = T2[:3, 3]; q = R.from_matrix(T2[:3,:3]).as_quat()
print(f"  t: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]   q: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")

print("=== Variant 3: (RAW) @ (camlink_optical) ===")
T3 = T_RAW @ T_camlink_optical
t = T3[:3, 3]; q = R.from_matrix(T3[:3,:3]).as_quat()
print(f"  t: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]   q: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")

print("=== Variant 4: inv(RAW) @ (camlink_optical) ===")
T4 = np.linalg.inv(T_RAW) @ T_camlink_optical
t = T4[:3, 3]; q = R.from_matrix(T4[:3,:3]).as_quat()
print(f"  t: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]   q: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
