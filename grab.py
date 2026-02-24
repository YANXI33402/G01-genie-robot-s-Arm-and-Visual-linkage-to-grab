#!/usr/bin/env python3
import time
import numpy as np
import cv2

from a2d_sdk.robot import RobotController
from a2d_sdk.robot import RobotDds
from sandbag_detection import SandbagDetector


# 全局变量：控制抓取
grab_triggered = False

# 右臂初始化姿势（视觉检测前的待机位）
RIGHT_ARM_INIT_POSE = [
    -1.8260728120803833,
    0.3092663288116455,
    -1.9050687551498413,
    -1.243626594543457,
    -2.879633903503418,
    -0.6965690851211548,
    0.26968973875045776,
]

# 抓取时末端目标姿态（四元数 x,y,z,w）
RIGHT_ARM_TARGET_ORIENTATION = np.array(
    [0.9119376949984299,0.03257955506292911,0.27278178419286137, -0.3047922427579362],
    dtype=np.float64,
)
RIGHT_ARM_TARGET_ORIENTATION /= np.linalg.norm(RIGHT_ARM_TARGET_ORIENTATION)


def on_button_click(event, x, y, flags, param):
    """鼠标点击回调"""
    global grab_triggered
    if event == cv2.EVENT_LBUTTONDOWN:
        grab_triggered = True
        print(">>> 抓取指令已触发！")


def poll_state(getter, length, name, timeout=2.0, interval=0.1):
    """等待指定关节状态就绪"""
    deadline = time.time() + timeout
    last_vals = None
    while time.time() < deadline:
        vals, _ = getter()
        last_vals = vals
        try:
            actual_len = len(vals)
        except Exception:
            actual_len = None
        if (actual_len == length) and all(_is_number(v) for v in vals):
            print(f"✓ {name} 就绪")
            return list(vals)
        time.sleep(interval)
    raise RuntimeError(f"{name} 在 {timeout}s 内未就绪, 最后 vals={last_vals!r}")


def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def main():
    global grab_triggered

    # ---------- 初始化 SDK ----------
    rc = RobotController()
    rd = RobotDds()
    time.sleep(2.0)

    # ---------- 等待状态就绪 ----------
    poll_state(rd.head_joint_states, length=2, name="head_joint_states")
    poll_state(rd.waist_joint_states, length=2, name="waist_joint_states")
    poll_state(rd.arm_joint_states, length=14, name="arm_joint_states")

    # ---------- 初始化姿势：右臂抬到设定位置 ----------
    print(">>> 右臂移动到初始化姿势...")
    rd.move_gripper([1,0.2])
    time.sleep(1)
    init_lifetime = 10.0
    rc.set_joint_position_control(init_lifetime, {"right_arm":RIGHT_ARM_INIT_POSE})
    time.sleep(init_lifetime + 0.5)
    # rc.set_joint_position_control(init_lifetime, {"waist_pitch":[0.0], "waist_lift":[0.0]})
    # time.sleep(init_lifetime + 0.5)
    print(">>> 右臂初始化姿势到位")

    # ---------- 初始化视觉 ----------
    detector = SandbagDetector()
    cv2.namedWindow("result")
    cv2.setMouseCallback("result", on_button_click)
    cv2.setWindowTitle("result", "result - 第一次点击预览，第二次点击确认抓取")

    print(">>> 开始实时检测沙袋位置")
    print(">>> 第一次点击：预览 base_link 目标位置；第二次点击：确认抓取")

    # 相机 → base_link 外参
    R_cam2base = np.array([
        [-0.003395,  0.000042,  0.999994],
        [-0.999988,  0.003437, -0.003395],
        [-0.003437, -0.999994,  0.00003]
    ], dtype=np.float64)
    # 平移：相机原点在 base_link 下的坐标（Z 已 +5cm 修正末端偏低）
    t_cam2base = np.array([0.10149, -0.10593, 0.804303], dtype=np.float64)

    # ========== 阶段1：等待第一次点击（预览） ==========
    while True:
        if detector.acquire_images():
            detector.process()
            if detector.has_last:
                print(f">>> [实时] 相机: X={detector.Xf:.3f}, Y={detector.Yf:.3f}, Z={detector.Zf:.3f}")
            if grab_triggered:
                print(">>> 第一次点击，开始计算 base_link 目标...")
                # 获取当前检测到的位置
                p_cam_check = np.array([detector.Xf, detector.Yf, detector.Zf], dtype=np.float64)
                p_base_check = R_cam2base @ p_cam_check + t_cam2base
                p_base_check[2] += 0.05
                
                # 安全检查：y值应在[-0.3, 0.3]范围内
                y_value = p_base_check[1]
                z_value = p_base_check[2]
                if y_value < -0.3 or y_value > 0.3 or z_value < 0.4:
                    print(f">>> ⚠️  安全限位检查失败：Y={y_value:.3f} 不在允许范围[-0.3, 0.3]内，请重新点击")
                    grab_triggered = False
                    continue
                else:
                    print(f">>> ✓ 安全限位检查通过：Y={y_value:.3f} 在允许范围[-0.3, 0.3]内")
                    break
        time.sleep(0.01)

    # ========== 阶段2：持续输出 base_link 目标，等待第二次点击 ==========
    print(">>> 持续输出 base_link 目标位置，第二次点击确认抓取")
    grab_triggered = False

    while True:
        if detector.acquire_images():
            detector.process()
            if detector.has_last:
                p_cam = np.array([detector.Xf, detector.Yf, detector.Zf], dtype=np.float64)
                p_base = R_cam2base @ p_cam + t_cam2base
                p_base[2] += 0.05
                print(f">>> [base_link 目标] X={p_base[0]:.3f}, Y={p_base[1]:.3f}, Z={p_base[2]:.3f}")
            if grab_triggered:
                print(">>> 第二次点击确认，执行末端绝对位姿控制...")
                p_cam = np.array([detector.Xf, detector.Yf, detector.Zf], dtype=np.float64)
                break
        time.sleep(0.01)

    # ========== 阶段3：末端绝对位姿控制 ==========
    p_base = R_cam2base @ p_cam + t_cam2base
    #p_base[2] += 0.05
    print(f">>> 目标位姿 (base_link): position={p_base.tolist()}")

    # SDK 要求：control_group 为可迭代（如列表）；位姿为 x,y,z,qx,qy,qz,qw 平铺字典
    right_pose = {
        "x": float(p_base[0]),
        "y": float(p_base[1]),
        "z": float(p_base[2]),
        "qx": float(RIGHT_ARM_TARGET_ORIENTATION[0]),
        "qy": float(RIGHT_ARM_TARGET_ORIENTATION[1]),
        "qz": float(RIGHT_ARM_TARGET_ORIENTATION[2]),
        "qw": float(RIGHT_ARM_TARGET_ORIENTATION[3]),
    }
    pose_lifetime = 5.0
    print(">>> 末端移动到目标位姿")
    rc.set_end_effector_pose_control(
        pose_lifetime,
        ["right_arm"],
        left_pose=None,
        right_pose=right_pose,
    )
    print(">>> 等待 3s 机械臂到位")
    time.sleep(3)
    print(">>> 夹爪合拢")
    rd.move_gripper([1, 0.9])
    print(">>> 等待 3s")
    time.sleep(3)

    # 夹住后末端向上移动 5cm（base_link 下 Z 轴向上）
    p_base_lift = p_base + np.array([0, 0, 0.05], dtype=np.float64)
    right_pose_lift = {
        "x": float(p_base_lift[0]),
        "y": float(p_base_lift[1]),
        "z": float(p_base_lift[2]),
        "qx": float(RIGHT_ARM_TARGET_ORIENTATION[0]),
        "qy": float(RIGHT_ARM_TARGET_ORIENTATION[1]),
        "qz": float(RIGHT_ARM_TARGET_ORIENTATION[2]),
        "qw": float(RIGHT_ARM_TARGET_ORIENTATION[3]),
    }
    print(">>> 末端向上移动 5cm")
    rc.set_end_effector_pose_control(
        pose_lifetime,
        ["right_arm"],
        left_pose=None,
        right_pose=right_pose_lift,
    )
    print(">>> 等待末端到位")
    time.sleep(pose_lifetime + 0.5)

    # 左移 5cm（base_link 下 Y 轴正方向为左）
    p_base_left = p_base_lift + np.array([0, -0.08, 0], dtype=np.float64)
    right_pose_left = {
        "x": float(p_base_left[0]),
        "y": float(p_base_left[1]),
        "z": float(p_base_left[2]),
        "qx": float(RIGHT_ARM_TARGET_ORIENTATION[0]),
        "qy": float(RIGHT_ARM_TARGET_ORIENTATION[1]),
        "qz": float(RIGHT_ARM_TARGET_ORIENTATION[2]),
        "qw": float(RIGHT_ARM_TARGET_ORIENTATION[3]),
    }
    print(">>> 末端左移 5cm")
    rc.set_end_effector_pose_control(
        pose_lifetime,
        ["right_arm"],
        left_pose=None,
        right_pose=right_pose_left,
    )
    print(">>> 等待 3s")
    time.sleep(3)

    # 向下移动 5cm，释放夹爪，等待 3s
    p_base_down = p_base_left + np.array([0, 0, -0.05], dtype=np.float64)
    right_pose_down = {
        "x": float(p_base_down[0]),
        "y": float(p_base_down[1]),
        "z": float(p_base_down[2]),
        "qx": float(RIGHT_ARM_TARGET_ORIENTATION[0]),
        "qy": float(RIGHT_ARM_TARGET_ORIENTATION[1]),
        "qz": float(RIGHT_ARM_TARGET_ORIENTATION[2]),
        "qw": float(RIGHT_ARM_TARGET_ORIENTATION[3]),
    }
    print(">>> 末端下移 5cm")
    rc.set_end_effector_pose_control(
        pose_lifetime,
        ["right_arm"],
        left_pose=None,
        right_pose=right_pose_down,
    )
    print(">>> 等待末端到位")
    time.sleep(pose_lifetime + 0.5)
    print(">>> 释放夹爪")
    rd.move_gripper([1, 0.2])
    print(">>> 等待 3s")
    time.sleep(3)

    # 释放后末端向上移动 5cm，再恢复初始位姿
    p_base_up = p_base_down + np.array([0, 0, 0.1], dtype=np.float64)
    right_pose_up = {
        "x": float(p_base_up[0]),
        "y": float(p_base_up[1]),
        "z": float(p_base_up[2]),
        "qx": float(RIGHT_ARM_TARGET_ORIENTATION[0]),
        "qy": float(RIGHT_ARM_TARGET_ORIENTATION[1]),
        "qz": float(RIGHT_ARM_TARGET_ORIENTATION[2]),
        "qw": float(RIGHT_ARM_TARGET_ORIENTATION[3]),
    }
    print(">>> 末端向上移动 5cm")
    rc.set_end_effector_pose_control(
        pose_lifetime,
        ["right_arm"],
        left_pose=None,
        right_pose=right_pose_up,
    )
    print(">>> 等待末端到位")
    time.sleep(pose_lifetime + 0.5)

    # 恢复初始位姿
    print(">>> 右臂恢复初始化位姿")
    rc.set_joint_position_control(init_lifetime, {"right_arm": RIGHT_ARM_INIT_POSE})
    time.sleep(init_lifetime + 0.5)
    print(">>> 抓取流程结束")

    # 继续显示检测，按 'q' 退出
    print(">>> 按 'q' 退出")
    try:
        while True:
            if detector.acquire_images():
                detector.process()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:q
    finally:
        detector.camera.close()
        cv2.destroyAllWindows()
        rd.shutdown()


if __name__ == "__main__":
    main()
