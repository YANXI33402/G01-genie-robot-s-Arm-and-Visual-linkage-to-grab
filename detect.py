# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""读取Isaac Lab中两个坐标系之间的4x4齐次变换矩阵

功能：获取机器人两个body之间的齐次变换矩阵（使用IsaacLab标准API）

Usage:
    ./isaaclab.sh -p g1_vision/detect.py
"""

import argparse

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="获取Isaac Lab中两个坐标系之间的变换矩阵")
parser.add_argument("--robot_name", type=str, default="G1bot", help="机器人名称（在场景中的key）")
parser.add_argument("--base_body", type=str, default="base_link", help="基坐标系body名称（默认：base_link）")
parser.add_argument("--target_body", type=str, default="head_link2", help="目标坐标系body名称（默认：Head_Camera）")
parser.add_argument("--list_bodies", action="store_true", help="列出所有可用的body名称")
parser.add_argument("--list_joints", action="store_true", help="列出所有可用的关节名称")
parser.add_argument("--camera_to_head", action="store_true", help="计算Head_Camera相对于head_link2的变换矩阵")
parser.add_argument("--camera_to_base", action="store_true", help="计算Head_Camera相对于base_link的变换矩阵")
parser.add_argument("--transform_point", type=float, nargs=3, metavar=("X", "Y", "Z"), help="将Head_Camera系下的点转换到base_link系（格式：--transform_point x y z）")
parser.add_argument("--use_default_point", action="store_true", help="使用代码中设置的默认点进行转换（默认点可在代码中修改）")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# ==================== 默认点配置（可以在这里直接修改） ====================
# 如果你想在代码中直接指定原始点，修改下面的值即可
# 这个点是在 Head_Camera 坐标系下的坐标 [x, y, z]
# 注意：这个值会在 main() 函数中初始化（因为需要先导入 numpy）
# ============================================================================


# 如果需要加载机器人，可以从add_new_robot.py导入G1_CONFIG
# 示例：
# from g1_vision.add_new_robot import G1_CONFIG
# 然后在TransformSceneCfg中添加：
# G1bot: ArticulationCfg = G1_CONFIG.replace(prim_path="{ENV_REGEX_NS}/G1bot")
G1_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lxh/IsaacLab/assets/G1_omnipicker/G1_omnipicker.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "idx01_body_joint1": 0.0,
            "idx02_body_joint2": 0.0,
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.3491,
            "idx21_arm_l_joint1": -0.6696,
            "idx61_arm_r_joint1": 0.6699,
            "idx22_arm_l_joint2": 0.201,
            "idx62_arm_r_joint2": -0.201,
            "idx23_arm_l_joint3": 0.27,
            "idx63_arm_r_joint3": -0.27,
            "idx24_arm_l_joint4": -1.2,
            "idx64_arm_r_joint4": 1.2,
            "idx25_arm_l_joint5": 0.8,
            "idx65_arm_r_joint5": -0.8,
            "idx26_arm_l_joint6": 1.57,
            "idx66_arm_r_joint6": -1.57,
            "idx27_arm_l_joint7": -0.18,
            "idx67_arm_r_joint7": 0.18,
            "idx31_gripper_l_inner_joint1": 0.4,  # Increased to open more
            "idx41_gripper_l_outer_joint1": 0.4,  # Increased to open more
            "idx71_gripper_r_inner_joint1": 0.4,  # Increased to open more
            "idx81_gripper_r_outer_joint1": 0.4,  # Increased to open more
            "idx32_gripper_l_inner_joint3": 0.1,
            "idx42_gripper_l_outer_joint3": 0.1,
            "idx72_gripper_r_inner_joint3": -0.1,
            "idx82_gripper_r_outer_joint3": 0.1,
            "idx33_gripper_l_inner_joint4": 0.0,
            "idx43_gripper_l_outer_joint4": 0.0,
            "idx73_gripper_r_inner_joint4": 0.0,
            "idx83_gripper_r_outer_joint4": 0.0,
            "idx54_gripper_l_inner_joint0": 0.0,
            "idx53_gripper_l_outer_joint0": 0.0,
            "idx94_gripper_r_inner_joint0": 0.0,
            "idx93_gripper_r_outer_joint0": 0.0,
        },
    ),
    actuators={
        "body_joints": ImplicitActuatorCfg(
            joint_names_expr=["idx0[1-2]_body_joint.*"],
            stiffness=100000.0,
            damping=100.0,
            velocity_limit_sim=None,
        ),
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["idx[26][1-7]_arm_[lr]_joint.*"],
            stiffness=None,
            damping=None,
            velocity_limit_sim=None,
            effort_limit_sim=None,
        ),
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*gripper.*"],
            stiffness=None,
            damping=None,
            velocity_limit_sim=None,
            effort_limit_sim=None,
        ),
        "head_joints": ImplicitActuatorCfg(
            joint_names_expr=["idx1[1-2]_head_joint.*"],
            stiffness=None,
            damping=None,
            velocity_limit_sim=None,
            effort_limit_sim=None,
        ),
    },
)


@configclass
class TransformSceneCfg(InteractiveSceneCfg):
    """场景配置：用于加载机器人并获取变换矩阵"""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/lxh/IsaacLab/assets/scene/table.usd",
            scale=(0.004, 0.006, 0.008),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                rigid_body_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    cube = AssetBaseCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(2.0, 0.0, 0.0),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.83), rot=(0, 0, 0, 1)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    G1bot: ArticulationCfg = G1_CONFIG.replace(prim_path="{ENV_REGEX_NS}/G1bot")

    # Head_Camera传感器配置
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/G1bot/head_link2/Head_Camera",
        update_period=0.0,  # 每帧更新
        height=480,
        width=640,
        data_types=[],  # 空列表，只需要位姿数据，不需要图像数据
        spawn=None,  # 相机已经在机器人USD中存在，不需要再生成
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    )


def list_all_bodies(scene: InteractiveScene, robot_name: str) -> None:
    """
    列出机器人所有可用的body名称
    
    参数:
        scene: InteractiveScene对象
        robot_name: 机器人在scene中的key名称
    """
    # 使用 try-except 安全地检查实体是否存在
    try:
        robot = scene[robot_name]
    except KeyError:
        available_entities = list(scene.keys())
        print(f"错误：场景中不存在名为 '{robot_name}' 的机器人！")
        print(f"可用的场景对象：{available_entities}")
        return
    
    print(f"\n机器人 '{robot_name}' 的所有body名称：")
    print("=" * 60)
    if hasattr(robot.data, 'body_names'):
        for i, body_name in enumerate(robot.data.body_names):
            print(f"  [{i:3d}] {body_name}")
    else:
        print("  无法获取body名称列表（可能需要先执行一步仿真）")
    print("=" * 60)


def list_all_joints(scene: InteractiveScene, robot_name: str) -> None:
    """
    列出机器人所有可用的关节名称
    
    参数:
        scene: InteractiveScene对象
        robot_name: 机器人在scene中的key名称
    """
    # 使用 try-except 安全地检查实体是否存在
    try:
        robot = scene[robot_name]
    except KeyError:
        available_entities = list(scene.keys())
        print(f"错误：场景中不存在名为 '{robot_name}' 的机器人！")
        print(f"可用的场景对象：{available_entities}")
        return
    
    print(f"\n机器人 '{robot_name}' 的所有关节名称：")
    print("=" * 60)
    if hasattr(robot.data, 'joint_names'):
        for i, joint_name in enumerate(robot.data.joint_names):
            joint_pos = robot.data.joint_pos[0, i].item() if robot.data.joint_pos.shape[0] > 0 else 0.0
            print(f"  [{i:3d}] {joint_name:40s} 位置: {joint_pos:8.4f}")
    else:
        print("  无法获取关节名称列表（可能需要先执行一步仿真）")
    print("=" * 60)


def get_transform_matrix_between_two_frames(
    scene: InteractiveScene,
    robot_name: str,
    base_body_name: str,
    target_body_name: str,
    env_idx: int = 0,
) -> np.ndarray:
    """
    获取两个坐标系之间的4x4齐次变换矩阵（IsaacLab标准方式）
    
    参数:
        scene: InteractiveScene对象
        robot_name: 机器人在scene中的key名称
        base_body_name: 基坐标系body名称
        target_body_name: 目标坐标系body名称
        env_idx: 环境索引（默认0，用于多环境场景）
    
    返回:
        4x4齐次变换矩阵（numpy数组），表示target相对于base的变换
    """
    # 获取机器人对象（使用 try-except 安全地检查实体是否存在）
    try:
        robot = scene[robot_name]
    except KeyError:
        available_entities = list(scene.keys())
        raise ValueError(
            f"错误：场景中不存在名为 '{robot_name}' 的机器人！\n"
            f"可用的场景对象：{available_entities}"
        )
    
    # 配置base body实体
    base_body_cfg = SceneEntityCfg(robot_name, body_names=[base_body_name])
    base_body_cfg.resolve(scene)
    
    # 配置target body实体
    target_body_cfg = SceneEntityCfg(robot_name, body_names=[target_body_name])
    target_body_cfg.resolve(scene)
    
    # 检查body是否存在，如果不存在则列出所有可用的body名称
    if len(base_body_cfg.body_ids) == 0:
        print(f"\n错误：找不到名为 '{base_body_name}' 的body！")
        list_all_bodies(scene, robot_name)
        raise ValueError(f"找不到名为 '{base_body_name}' 的body！请检查body名称是否正确。")
    
    if len(target_body_cfg.body_ids) == 0:
        print(f"\n错误：找不到名为 '{target_body_name}' 的body！")
        list_all_bodies(scene, robot_name)
        raise ValueError(f"找不到名为 '{target_body_name}' 的body！请检查body名称是否正确。")
    
    # 获取两个body在世界坐标系下的位姿
    # robot.data.body_pose_w 形状: [num_envs, num_bodies, 7] (x, y, z, qx, qy, qz, qw)
    base_body_pose_w = robot.data.body_pose_w[env_idx, base_body_cfg.body_ids[0]]  # [7]
    target_body_pose_w = robot.data.body_pose_w[env_idx, target_body_cfg.body_ids[0]]  # [7]
    
    # 提取位置和四元数
    base_pos_w = base_body_pose_w[:3]  # [x, y, z]
    base_quat_w = base_body_pose_w[3:7]  # [qx, qy, qz, qw]
    
    target_pos_w = target_body_pose_w[:3]  # [x, y, z]
    target_quat_w = target_body_pose_w[3:7]  # [qx, qy, qz, qw]
    
    # 确保数据是torch tensor格式（IsaacLab使用torch）
    # robot.data返回的是torch tensor，但为了安全起见进行检查
    if not isinstance(base_pos_w, torch.Tensor):
        base_pos_w = torch.from_numpy(base_pos_w).to(robot.device)
        base_quat_w = torch.from_numpy(base_quat_w).to(robot.device)
        target_pos_w = torch.from_numpy(target_pos_w).to(robot.device)
        target_quat_w = torch.from_numpy(target_quat_w).to(robot.device)
    
    # 计算target相对于base的变换（使用IsaacLab标准方法）
    # subtract_frame_transforms: 计算target相对于base的位姿
    # 输入：base在世界坐标系下的位姿，target在世界坐标系下的位姿
    # 输出：target在base坐标系下的位姿
    relative_pos, relative_quat = subtract_frame_transforms(
        base_pos_w.unsqueeze(0), base_quat_w.unsqueeze(0),
        target_pos_w.unsqueeze(0), target_quat_w.unsqueeze(0)
    )
    # 转换为numpy数组
    relative_pos = relative_pos.squeeze(0).cpu().numpy()  # [3]
    relative_quat = relative_quat.squeeze(0).cpu().numpy()  # [4]
    
    # 将四元数转换为旋转矩阵
    # IsaacLab使用[x, y, z, w]格式，scipy的Rotation.from_quat也使用[x, y, z, w]格式
    from scipy.spatial.transform import Rotation
    # relative_quat格式: [qx, qy, qz, qw]
    rotation = Rotation.from_quat(relative_quat)
    rotation_matrix = rotation.as_matrix()
    
    # 构建4x4齐次变换矩阵
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = relative_pos
    
    # 格式化输出结果
    print(f"\n成功获取 {base_body_name} → {target_body_name} 的变换矩阵：")
    print("=" * 60)
    print(np.round(transform_matrix, 6))
    print("=" * 60)
    print(f"平移向量（x, y, z）：{np.round(transform_matrix[:3, 3], 6)}")
    print(f"旋转矩阵：\n{np.round(transform_matrix[:3, :3], 6)}")
    print(f"四元数（qx, qy, qz, qw）：{np.round(relative_quat, 6)}")
    
    return transform_matrix


def get_camera_transform_relative_to_body(
    scene: InteractiveScene,
    robot_name: str,
    parent_body_name: str,
    env_idx: int = 0,
) -> np.ndarray:
    """
    获取相机相对于父body的变换矩阵（使用isaaclab.sensor API）
    
    参数:
        scene: InteractiveScene对象
        robot_name: 机器人在scene中的key名称
        parent_body_name: 父body名称（如 head_link2）
        env_idx: 环境索引（默认0）
    
    返回:
        4x4齐次变换矩阵（numpy数组），表示camera相对于parent_body的变换
    """
    from scipy.spatial.transform import Rotation

    try:
        robot = scene[robot_name]
    except KeyError:
        available_entities = list(scene.keys())
        raise ValueError(
            f"错误：场景中不存在名为 '{robot_name}' 的机器人！\n"
            f"可用的场景对象：{available_entities}"
        )
    
    try:
        head_camera = scene["head_camera"]
    except KeyError:
        raise ValueError(
            f"错误：场景中不存在名为 'head_camera' 的相机传感器！\n"
            f"请在场景配置中添加相机配置。"
        )
    
    # 获取父body的位姿
    parent_body_cfg = SceneEntityCfg(robot_name, body_names=[parent_body_name])
    parent_body_cfg.resolve(scene)
    
    if len(parent_body_cfg.body_ids) == 0:
        raise ValueError(f"找不到名为 '{parent_body_name}' 的body！")
    
    # 获取父body在世界坐标系下的位姿
    # body_pose_w 格式: [x, y, z, w, x, y, z] (前3个是位置，后4个是四元数)
    parent_body_pose_w = robot.data.body_pose_w[env_idx, parent_body_cfg.body_ids[0]]  # [7]
    parent_pos_w = parent_body_pose_w[:3]  # [x, y, z]
    parent_quat_w = parent_body_pose_w[3:7]  # [w, x, y, z]
    
    # 使用isaaclab.sensor API获取相机在世界坐标系下的位置和姿态
    # pos_w: 相机在世界坐标系下的位置 [x, y, z]
    # quat_w_world: 相机在世界坐标系下的四元数 [w, x, y, z] (world约定，forward +X, up +Z)
    # quat_w_ros: 相机在世界坐标系下的四元数 [w, x, y, z] (ROS约定，forward +Z, up -Y)
    # quat_w_opengl: 相机在世界坐标系下的四元数 [w, x, y, z] (OpenGL约定，forward -Z, up +Y)
    camera_pos_w = head_camera.data.pos_w[env_idx]  # [x, y, z]
    camera_quat_w_world = head_camera.data.quat_w_world[env_idx]  # [w, x, y, z]
    
    # 输出相机在世界坐标系下的位姿信息
    print(f"\n[INFO]: Head_Camera 在世界坐标系下的位姿：")
    print(f"  位置 (pos_w):           {camera_pos_w.cpu().numpy()}")
    print(f"  四元数 World (quat_w_world):  {camera_quat_w_world.cpu().numpy()}")
    print(f"  四元数 ROS (quat_w_ros):      {head_camera.data.quat_w_ros[env_idx].cpu().numpy()}")
    print(f"  四元数 OpenGL (quat_w_opengl): {head_camera.data.quat_w_opengl[env_idx].cpu().numpy()}")
    
    # 输出父body在世界坐标系下的位姿信息
    print(f"\n[INFO]: {parent_body_name} 在世界坐标系下的位姿：")
    print(f"  位置: {parent_pos_w.cpu().numpy()}")
    print(f"  四元数: {parent_quat_w.cpu().numpy()}")
    
    # 计算相机相对于父body的变换
    # subtract_frame_transforms 使用 (w, x, y, z) 格式的四元数
    relative_pos, relative_quat = subtract_frame_transforms(
        parent_pos_w.unsqueeze(0), parent_quat_w.unsqueeze(0),
        camera_pos_w.unsqueeze(0), camera_quat_w_world.unsqueeze(0)
    )
    
    # 转换为numpy数组
    relative_pos = relative_pos.squeeze(0).cpu().numpy()  # [x, y, z]
    relative_quat = relative_quat.squeeze(0).cpu().numpy()  # [w, x, y, z]
    
    # 构建4x4齐次变换矩阵
    # scipy的Rotation.from_quat使用 [x, y, z, w] 格式
    rotation = Rotation.from_quat(np.array([
        relative_quat[1], relative_quat[2], relative_quat[3], relative_quat[0]
    ]))
    rotation_matrix = rotation.as_matrix()
    
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = relative_pos
    
    print(f"\n[INFO]: Head_Camera 相对于 {parent_body_name} 的变换矩阵：")
    print("=" * 60)
    print(np.round(transform_matrix, 6))
    print("=" * 60)
    print(f"平移向量 (x, y, z): {np.round(transform_matrix[:3, 3], 6)}")
    print(f"旋转矩阵:\n{np.round(transform_matrix[:3, :3], 6)}")
    print(f"四元数 (w, x, y, z): {np.round(relative_quat, 6)}")
    
    return transform_matrix


def transform_point_from_camera_to_base(
    scene: InteractiveScene,
    robot_name: str,
    point_in_camera: np.ndarray,
    env_idx: int = 0,
) -> np.ndarray:
    """
    将Head_Camera坐标系下的点转换到base_link坐标系下
    
    参数:
        scene: InteractiveScene对象
        robot_name: 机器人在scene中的key名称
        point_in_camera: Head_Camera坐标系下的点 [x, y, z] 或 [x, y, z, 1] (齐次坐标)
        env_idx: 环境索引（默认0）
    
    返回:
        base_link坐标系下的点 [x, y, z]
    """
    try:
        robot = scene[robot_name]
    except KeyError:
        available_entities = list(scene.keys())
        raise ValueError(
            f"错误：场景中不存在名为 '{robot_name}' 的机器人！\n"
            f"可用的场景对象：{available_entities}"
        )
    
    # 1. 获取 Head_Camera 相对于 head_link2 的变换
    T_camera_to_head = get_camera_transform_relative_to_body(
        scene, robot_name, "head_link2", env_idx=env_idx
    )
    
    # 2. 获取 head_link2 相对于 base_link 的变换
    T_head_to_base = get_transform_matrix_between_two_frames(
        scene, robot_name, "base_link", "head_link2", env_idx
    )
    
    # 3. 组合变换：T_camera_to_base = T_head_to_base * T_camera_to_head
    T_camera_to_base = T_head_to_base @ T_camera_to_head
    
    # 4. 转换点坐标
    if point_in_camera.shape[0] == 3:
        # 如果不是齐次坐标，转换为齐次坐标
        point_homogeneous = np.append(point_in_camera, 1.0)
    else:
        point_homogeneous = point_in_camera
    
    # 应用变换矩阵
    point_in_base_homogeneous = T_camera_to_base @ point_homogeneous
    point_in_base = point_in_base_homogeneous[:3]
    
    print(f"\n坐标变换结果：")
    print("=" * 60)
    print(f"Head_Camera 系下的点: {np.round(point_in_camera[:3], 6)}")
    print(f"base_link 系下的点:   {np.round(point_in_base, 6)}")
    print("=" * 60)
    print(f"\n使用的变换矩阵链：")
    print(f"  T_camera_to_head (Head_Camera → head_link2):")
    print(np.round(T_camera_to_head, 6))
    print(f"\n  T_head_to_base (head_link2 → base_link):")
    print(np.round(T_head_to_base, 6))
    print(f"\n  T_camera_to_base (Head_Camera → base_link):")
    print(np.round(T_camera_to_base, 6))
    
    return point_in_base


def main() -> None:
    """主函数：初始化场景并获取变换矩阵"""
    # ==================== 默认点配置（可以在这里直接修改） ====================
    # 如果你想在代码中直接指定原始点，修改下面的值即可
    # 这个点是在 Head_Camera 坐标系下的坐标 [x, y, z]
    # 示例：相机前方1米处
    DEFAULT_POINT_IN_CAMERA = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # ============================================================================
    
    # 创建仿真上下文
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0/60.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.0, 0.0, 2.5), (0.0, 0.0, 0.5))
    
    # 创建场景（注意：这里假设机器人已经在场景中）
    # 如果机器人不在场景中，需要先加载机器人配置
    scene_cfg = TransformSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 重置仿真
    sim.reset()
    scene.reset()

    # 手动设置机器人关节角度（从G1_CONFIG中读取）
    try:
        robot = scene[args_cli.robot_name]
        # 获取G1_CONFIG中定义的关节角度
        joint_pos = G1_CONFIG.init_state.joint_pos
        # 将关节位置设置到机器人
        for joint_name, target_pos in joint_pos.items():
            # 查找关节索引
            if joint_name in robot.data.joint_names:
                joint_idx = robot.data.joint_names.index(joint_name)
                robot.data.default_joint_pos[0, joint_idx] = target_pos

        # 写入到仿真
        scene.write_data_to_sim()

        # 执行多步仿真让机器人稳定到目标位置
        for _ in range(10):
            sim.step()
            scene.update(sim.get_physics_dt())

        print(f"[INFO]: 已设置机器人关节角度到 G1_CONFIG 中定义的值")
    except Exception as e:
        print(f"[WARN]: 设置关节角度时出错: {e}")
        # 如果设置失败，仍然继续执行后续步骤

    # 执行一步仿真以确保数据更新
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    
    print("[INFO]: 场景初始化完成...")
    print(f"[INFO]: 查找机器人 '{args_cli.robot_name}'...")
    
    # 如果只是列出body名称，则执行后退出
    if args_cli.list_bodies:
        try:
            list_all_bodies(scene, args_cli.robot_name)
            return None
        except Exception as e:
            print(f"\n[ERROR]: 列出body名称时发生错误：{e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 如果只是列出关节名称，则执行后退出
    if args_cli.list_joints:
        try:
            list_all_joints(scene, args_cli.robot_name)
            return None
        except Exception as e:
            print(f"\n[ERROR]: 列出关节名称时发生错误：{e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 如果计算Head_Camera相对于head_link2的变换矩阵
    if args_cli.camera_to_head:
        try:
            print("[INFO]: 计算 Head_Camera 相对于 head_link2 的变换矩阵...")

            transform_matrix = get_camera_transform_relative_to_body(
                scene, args_cli.robot_name, "head_link2", env_idx=0
            )

            if transform_matrix is not None:
                print("\n[INFO]: Head_Camera 相对于 head_link2 的变换矩阵获取成功！")
                return transform_matrix
            else:
                print("\n[ERROR]: 变换矩阵获取失败！")
                return None
        except Exception as e:
            print(f"\n[ERROR]: 计算变换矩阵时发生错误：{e}")
            import traceback
            traceback.print_exc()
            return None

    # 如果计算Head_Camera相对于base_link的变换矩阵
    if args_cli.camera_to_base:
        try:
            print("[INFO]: 计算 Head_Camera 相对于 base_link 的变换矩阵...")

            # 方法：计算 Head_Camera → head_link2 和 head_link2 → base_link，然后组合
            T_camera_to_head = get_camera_transform_relative_to_body(
                scene, args_cli.robot_name, "head_link2", env_idx=0
            )

            T_head_to_base = get_transform_matrix_between_two_frames(
                scene, args_cli.robot_name, "base_link", "head_link2", env_idx=0
            )

            # 组合变换：T_camera_to_base = T_head_to_base * T_camera_to_head
            # 注意：变换矩阵的组合顺序是反的
            T_camera_to_base = T_head_to_base @ T_camera_to_head

            print(f"\n[INFO]: Head_Camera 相对于 base_link 的变换矩阵获取成功！")
            print("=" * 60)
            print(np.round(T_camera_to_base, 6))
            print("=" * 60)
            print(f"平移向量 (x, y, z): {np.round(T_camera_to_base[:3, 3], 6)}")
            print(f"旋转矩阵:\n{np.round(T_camera_to_base[:3, :3], 6)}")

            return T_camera_to_base
        except Exception as e:
            print(f"\n[ERROR]: 计算变换矩阵时发生错误：{e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 如果转换点坐标
    if args_cli.transform_point is not None or args_cli.use_default_point:
        try:
            # 确定使用哪个点
            if args_cli.transform_point is not None:
                # 方式1：从命令行参数获取点
                point_in_camera = np.array(args_cli.transform_point, dtype=np.float64)
                print(f"[INFO]: 使用命令行参数指定的点")
            else:
                # 方式2：使用代码中设置的默认点
                # 默认点定义在 main() 函数开头，可以直接修改那里的值
                point_in_camera = DEFAULT_POINT_IN_CAMERA.copy()
                print(f"[INFO]: 使用代码中设置的默认点")
            
            print(f"[INFO]: Head_Camera 系下的原始点: {point_in_camera}")
            print(f"[INFO]: 开始转换到 base_link 系...")
            
            point_in_base = transform_point_from_camera_to_base(
                scene, args_cli.robot_name, point_in_camera, env_idx=0
            )
            
            if point_in_base is not None:
                print("\n[INFO]: 坐标转换成功！")
                print(f"[INFO]: 最终结果 - base_link 系下的点: {point_in_base}")
                return point_in_base
            else:
                print("\n[ERROR]: 坐标转换失败！")
                return None
        except Exception as e:
            print(f"\n[ERROR]: 坐标转换时发生错误：{e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 获取变换矩阵
    print(f"[INFO]: 计算 {args_cli.base_body} → {args_cli.target_body} 的变换矩阵...")
    try:
        transform_matrix = get_transform_matrix_between_two_frames(
            scene=scene,
            robot_name=args_cli.robot_name,
            base_body_name=args_cli.base_body,
            target_body_name=args_cli.target_body,
            env_idx=0
        )
        
        if transform_matrix is not None:
            print("\n[INFO]: 变换矩阵获取成功！")
            print(f"[INFO]: 变换矩阵表示 {args_cli.target_body} 相对于 {args_cli.base_body} 的位姿")
            return transform_matrix
        else:
            print("\n[ERROR]: 变换矩阵获取失败！")
            return None
            
    except Exception as e:
        print(f"\n[ERROR]: 获取变换矩阵时发生错误：{e}")
        print(f"\n提示：")
        print(f"  - 使用 --list_bodies 参数查看所有可用的body名称")
        print(f"  - 使用 --list_joints 参数查看所有可用的关节名称")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
    simulation_app.close()