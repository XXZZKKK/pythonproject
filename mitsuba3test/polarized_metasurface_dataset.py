"""
超表面正方体多视角数据集生成脚本 (适配 PANDORA 模型)
支持光谱和偏振渲染

本脚本生成 PANDORA 神经辐射场模型所需的完整数据集，包括:
1. RGB 图像 (.exr 格式)
2. 掩码图像 (.png 格式)
3. 法向量图 (.exr 格式) - 可选
4. 偏振信息 (Stokes 参数) - 可选
5. 相机参数 (cameras.npz)

作者: [xzk]
日期: 2026-02-04
版本: 2.0 - 支持光谱和偏振
"""

import mitsuba as mi
import numpy as np
import os
import imageio
from pathlib import Path

# ============================================================================
# 🔴 修改1: 使用光谱和偏振变体
# 原代码: mi.set_variant('cuda_ad_rgb')
# 新代码: 使用与PANDORA原作者相同的变体
# ============================================================================
mi.set_variant('cuda_ad_spectral_polarized')
# 说明:
# - scalar: 标量模式(非GPU加速,但更稳定)
# - spectral: 支持光谱渲染(波长相关)
# - polarized: 支持偏振光渲染(Stokes参数)

# ============================================================================
# 导入偏振相关工具
# 🔴 修改2: 添加偏振处理模块
# ============================================================================
import sys

sys.path.append('.')

ceramic_bsdf = {
    'type':'roughplastic',
    'diffuse_reflectance':{
        'type':'rgb',
        'value':[0.8,0.8,0.8]
    },
    'specular_reflectance':{
        'type':'rgb',
        'value':[1.0,1.0,1.0]
    },
    'alpha':0.05,
    'int_ior':1.5,
    'distribution':'beckmann'
}


# 如果你有偏振处理函数,可以导入
# from src.polarization import cues_from_stokes_stack_np
# 如果没有,我们提供简化版本
def extract_stokes_parameters(stokes_stack):
    """
    从 Stokes 参数栈中提取偏振信息

    Stokes 参数:
    - S0: 总光强
    - S1, S2, S3: 偏振状态参数

    参数:
        stokes_stack: Stokes参数栈 (H, W, 12) 或 (H, W, 16)
                     RGB三通道 × 4个Stokes参数 = 12通道
                     或包含其他AOV

    返回:
        dict: {
            's0': S0 强度 (H, W, 3),
            'dop': 偏振度 (H, W, 3),
            'aolp': 偏振角 (H, W, 3)
        }
    """
    # 提取 Stokes 参数
    # 假设格式: [S0_R, S0_G, S0_B, S1_R, S1_G, S1_B, S2_R, S2_G, S2_B, S3_R, S3_G, S3_B]
    s0 = stokes_stack[..., 0:3]  # S0: 总强度
    s1 = stokes_stack[..., 3:6]  # S1: 水平/垂直线偏振
    s2 = stokes_stack[..., 6:9]  # S2: ±45度线偏振
    s3 = stokes_stack[..., 9:12]  # S3: 圆偏振

    # 计算偏振度 (Degree of Polarization)
    # DOP = sqrt(S1^2 + S2^2 + S3^2) / S0
    dop = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) / (s0 + 1e-8)
    dop = np.clip(dop, 0, 1)

    # 计算偏振角 (Angle of Linear Polarization)
    # AOLP = 0.5 * atan2(S2, S1)
    aolp = 0.5 * np.arctan2(s2, s1)
    # 归一化到 [0, 1]
    aolp = (aolp + np.pi) / (2 * np.pi)

    return {
        's0': s0,
        'dop': dop,
        'aolp': aolp
    }


# ============================================================================
# 常量定义 (保持不变)
# ============================================================================

# 立方体的六个面及其属性
CUBE_FACES = {
    'front': {
        'center': [0, 0.5, 0],
        'normal': [0, 1, 0]
    },
    'back': {
        'center': [0, -0.5, 0],
        'normal': [0, -1, 0]
    },
    'right': {
        'center': [0.5, 0, 0],
        'normal': [1, 0, 0]
    },
    'left': {
        'center': [-0.5, 0, 0],
        'normal': [-1, 0, 0]
    },
    'top': {
        'center': [0, 0, 0.5],
        'normal': [0, 0, 1]
    },
    'bottom': {
        'center': [0, 0, -0.5],
        'normal': [0, 0, -1]
    }
}


# ============================================================================
# 辅助函数 (保持不变)
# ============================================================================

def calculate_optimal_spacing(grid_size, disk_radius):
    """
    计算圆盘在网格中的最优间距

    公式推导:
    设边距 g (gap), 网格大小 n, 圆盘半径 r, 面长度 L=1
    总长度: g + n×(2r) + (n-1)×g = (n+1)×g + 2nr = L
    解出: g = (L - 2nr) / (n+1)
    间距: spacing = g + 2r

    参数:
        grid_size: 网格大小 (n×n)
        disk_radius: 圆盘半径

    返回:
        dict: {'spacing': 间距, 'gap': 边距}
    """
    gap = (1 - 2 * grid_size * disk_radius) / (grid_size + 1)
    spacing = gap + 2 * disk_radius

    return {
        'spacing': spacing,
        'gap': gap,
    }


def get_disk_grid_positions(face_name, face_center, grid_size, disk_radius):
    """
    获取某个面上所有圆盘的位置

    参数:
        face_name: 面的名称 ('front', 'back', 'right', 'left', 'top', 'bottom')
        face_center: 面的中心坐标 [x, y, z]
        grid_size: 网格大小 (n×n)
        disk_radius: 圆盘半径

    返回:
        list: 所有圆盘的3D坐标列表
    """
    positions = []
    info = calculate_optimal_spacing(grid_size, disk_radius)
    spacing = info['spacing']

    # 网格中心索引
    center_index = (grid_size - 1) / 2

    for i in range(grid_size):
        for j in range(grid_size):
            # 计算相对于网格中心的偏移
            offset_i = (i - center_index) * spacing
            offset_j = (j - center_index) * spacing

            # 根据面的方向确定偏移方向
            if face_name in ['front', 'back']:
                # 前/后面: 平行于XZ平面
                offset_x = offset_i
                offset_y = 0
                offset_z = offset_j
            elif face_name in ['right', 'left']:
                # 右/左面: 平行于YZ平面
                offset_x = 0
                offset_y = offset_i
                offset_z = offset_j
            elif face_name in ['top', 'bottom']:
                # 顶/底面: 平行于XY平面
                offset_x = offset_i
                offset_y = offset_j
                offset_z = 0

            # 计算圆盘的最终位置
            offset = np.array([offset_x, offset_y, offset_z])
            disk_position = np.array(face_center) + offset
            positions.append(disk_position)

    return positions


def create_disk_on_face(face_name, disk_center, disk_radius, disk_height):
    """
    在指定面上创建一个圆盘(圆柱+顶盖)

    参数:
        face_name: 面的名称
        disk_center: 圆盘中心位置 [x, y, z]
        disk_radius: 圆盘半径
        disk_height: 圆盘高度
        material: 材质 ('Au' = 金)

    返回:
        tuple: (圆柱体字典, 圆盘顶盖字典)
    """
    # 根据面的方向确定圆盘的变换矩阵
    # 圆柱默认沿Z轴方向
    if face_name == 'front':
        # 前面: 法向量 +Y, 需要绕X轴旋转-90度
        transform = (mi.ScalarTransform4f.translate(disk_center.tolist()) @
                     mi.ScalarTransform4f.rotate([1, 0, 0], -90))
        transform_top = (mi.ScalarTransform4f.translate((disk_center + [0, disk_height, 0]).tolist()) @
                         mi.ScalarTransform4f.rotate([1, 0, 0], -90))

    elif face_name == 'back':
        # 后面: 法向量 -Y, 需要绕X轴旋转90度
        transform = (mi.ScalarTransform4f.translate(disk_center.tolist()) @
                     mi.ScalarTransform4f.rotate([1, 0, 0], 90))
        transform_top = (mi.ScalarTransform4f.translate((disk_center + [0, -disk_height, 0]).tolist()) @
                         mi.ScalarTransform4f.rotate([1, 0, 0], 90))

    elif face_name == 'right':
        # 右面: 法向量 +X, 需要绕Y轴旋转90度
        transform = (mi.ScalarTransform4f.translate(disk_center.tolist()) @
                     mi.ScalarTransform4f.rotate([0, 1, 0], 90))
        transform_top = (mi.ScalarTransform4f.translate((disk_center + [disk_height, 0, 0]).tolist()) @
                         mi.ScalarTransform4f.rotate([0, 1, 0], 90))

    elif face_name == 'left':
        # 左面: 法向量 -X, 需要绕Y轴旋转-90度
        transform = (mi.ScalarTransform4f.translate(disk_center.tolist()) @
                     mi.ScalarTransform4f.rotate([0, 1, 0], -90))
        transform_top = (mi.ScalarTransform4f.translate((disk_center + [-disk_height, 0, 0]).tolist()) @
                         mi.ScalarTransform4f.rotate([0, 1, 0], -90))

    elif face_name == 'top':
        # 顶面: 法向量 +Z, 不需要旋转
        transform = mi.ScalarTransform4f.translate(disk_center.tolist())
        transform_top = mi.ScalarTransform4f.translate((disk_center + [0, 0, disk_height]).tolist())

    elif face_name == 'bottom':
        # 底面: 法向量 -Z, 需要绕X轴旋转180度
        transform = (mi.ScalarTransform4f.translate(disk_center.tolist()) @
                     mi.ScalarTransform4f.rotate([1, 0, 0], 180))
        transform_top = (mi.ScalarTransform4f.translate((disk_center + [0, 0, -disk_height]).tolist()) @
                         mi.ScalarTransform4f.rotate([1, 0, 0], 180))

    # 创建圆柱体 (圆盘侧壁)
    cylinder_dict = {
        'type': 'cylinder',
        'p0': [0, 0, 0],
        'p1': [0, 0, disk_height],
        'radius': disk_radius,
        'to_world': transform,
        'bsdf': ceramic_bsdf
    }

    # 创建圆盘顶盖
    disk_top_dict = {
        'type': 'disk',
        'to_world': transform_top @ mi.ScalarTransform4f.scale([disk_radius, disk_radius, 1]),
        'bsdf': ceramic_bsdf
    }

    return cylinder_dict, disk_top_dict


def lookat_from_spherical(theta, phi, radius):
    """
    从球面坐标生成 look-at 相机矩阵

    参数:
        theta: 方位角 (弧度)
        phi: 仰角 (弧度)
        radius: 相机距离

    返回:
        list: [origin, target, up] 相机参数
    """
    # 球面坐标转笛卡尔坐标
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    origin = [x, y, z]
    target = [0, 0, 0]  # 看向原点
    up = [0, 0, 1]  # Z轴向上

    return [origin, target, up]


def lookat_to_world_matrix(
        lookat_params,
        fov_degrees,
        image_width,
        image_height):

    origin = np.array(lookat_params[0])
    target = np.array(lookat_params[1])
    up = np.array(lookat_params[2])

    # 计算相机坐标系的基向量
    forward = target - origin
    forward = forward / np.linalg.norm(forward)  # z轴 (指向目标)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)  # x轴

    up_new = np.cross(right, forward)  # y轴

    R = np.stack([right, up_new, -forward], axis=1)
    t = -R @ origin#平移矩阵
    E = np.concatenate([R,t[:,None]],axis=1)#外参矩阵

    focal_length = 0.5 * image_width / np.tan(0.5 * fov_degrees * np.pi / 180.0)
    K = np.array([
        [focal_length, 0.0,          image_width / 2.0],
        [0.0,          focal_length, image_height / 2.0],
        [0.0,          0.0,          1.0]
    ], dtype=np.float64)

    #投影矩阵
    P = K @ E

    # 7. 2D坐标原点转换 (第376-379行)
    flip_matrix = np.array([
        [1.0,  0.0, 0.0],
        [0.0, -1.0, image_height],
        [0.0,  0.0, 1.0]
    ], dtype=np.float64)
    P = flip_matrix @ P  # 3x4

    P = np.vstack([P, [0.0, 0.0, 0.0, 1.0]])  # 4x4

    # 9. 坐标系转换 Mitsuba → PANDORA (第388-394行)
    C = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    world_mat = P @ C  # 4x4
    scale_mat = np.eye(4, dtype=np.float64)

    return world_mat,scale_mat


# ============================================================================
# 🔴 修改3: 创建场景函数 - 添加偏振和光谱支持
# 主要变化:
# 1. 添加 render_stokes 参数控制是否渲染偏振
# 2. 添加 no_aovs 参数控制是否输出额外通道(法向量等)
# 3. 配置 integrator 以支持偏振渲染
# ============================================================================
def create_metasurface_scene_dict(
        disk_radius=0.05,
        disk_height=0.02,
        grid_size=4,
        camera_lookat=None,
        image_resolution=800,
        cam_fov=45,
        render_stokes=True,  # 🔴 新增: 是否渲染 Stokes 参数
        no_aovs=False,  # 🔴 新增: 是否禁用额外输出通道
):
    """
    创建超表面正方体场景字典 (不渲染)

    参数:
        disk_radius: 圆盘半径
        disk_height: 圆盘高度
        grid_size: 每个面的网格大小 (n×n)
        camera_lookat: 相机look-at参数 [origin, target, up]
        image_resolution: 图像分辨率
        cam_fov: 相机视场角(度)
        render_stokes: 是否渲染 Stokes 偏振参数
        no_aovs: 是否禁用额外输出通道(法向量等)

    返回:
        dict: Mitsuba场景字典
    """
    scene_dict = {
        'type': 'scene',
    }

    # 1. 添加主立方体 (超表面基底)
    scene_dict['main_cube'] = {
        'type': 'cube',
        'to_world': mi.ScalarTransform4f.scale([0.5, 0.5, 0.5]),
        'bsdf': ceramic_bsdf
    }

    # 2. 在每个面上添加圆盘网格
    disk_count = 0
    for face_name, face_info in CUBE_FACES.items():
        # 获取该面上所有圆盘的位置
        disk_positions = get_disk_grid_positions(
            face_name,
            face_info['center'],
            grid_size=grid_size,
            disk_radius=disk_radius
        )

        # 为每个位置创建圆盘
        for disk_index, disk_position in enumerate(disk_positions):
            cylinder, disk_top = create_disk_on_face(
                face_name,
                disk_position,
                disk_radius,
                disk_height,
            )

            # 添加到场景字典
            cylinder_name = f'disk_{face_name}_{disk_index}'
            disk_top_name = f'disk_top_{face_name}_{disk_index}'

            scene_dict[cylinder_name] = cylinder
            scene_dict[disk_top_name] = disk_top
            disk_count += 1

    print(f'已添加 {disk_count} 个圆盘到场景')

    # # 3. 添加地板
    # scene_dict['floor'] = {
    #     'type': 'rectangle',
    #     'to_world': (
    #             mi.ScalarTransform4f.translate([0, 0, -1]) @
    #             mi.ScalarTransform4f.scale([5, 5, 1])
    #     ),
    #     'bsdf': {
    #         'type': 'diffuse',
    #         'reflectance': {
    #             'type': 'rgb',
    #             'value': [0.5, 0.5, 0.5]
    #         }
    #     }
    # }

    scene_dict['sensor'] = {
        'type': 'perspective',
        'fov': cam_fov,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=camera_lookat[0],
            target=camera_lookat[1],
            up=camera_lookat[2]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': image_resolution,
            'height': image_resolution,
            'pixel_format': 'rgba',
            'component_format': 'float32',
            'rfilter': {
                'type': 'gaussian'
            }
        }
    }

    # 5. 添加主光源 (定向光)
    scene_dict['main_light'] = {
        'type': 'directional',
        'direction': [-1, -1, -1],
        'irradiance': {
            'type': 'rgb',
            'value': [3, 3, 3]
        }
    }

    # 6. 添加环境光
    scene_dict['ambient'] = {
        'type': 'constant',
        'radiance': {
            'type': 'rgb',
            'value': [0.3,0.3,0.3]
        }
    }


    # main_int = {
    #     'type': 'aov',
    #     'aovs':'nn:sh_normal',
    #     'child':{
    #         'type': 'path',  # 或 volpath
    #         'max_depth': 8,
    #         'hide_emitters': True
    #     }
    # }
        # if render_stokes:#render_normals_only:
        #     # 普通路径追踪
    if render_stokes == True:
        scene_dict['integrator']={
            'type':'stokes',
            'child': {
                'type':'path',
                'max_depth':8}
        }

    elif render_stokes == False:
        scene_dict['integrator'] = {
            'type': 'aov',
            'aovs':'nn:sh_normal',
        }


    return scene_dict


# ============================================================================
# 🔴 修改6: 渲染函数 - 处理偏振输出
# 主要变化:
# 1. 处理 Stokes 参数输出
# 2. 从 bitmap 中提取偏振信息
# 3. 生成掩码的方式调整
# ============================================================================
def render_single_view(scene_dict, samples_per_pixel, render_stokes=True):#需要渲染两次，先stokes再法线

    """
    渲染单个视角

    参数:
        scene: Mitsuba场景对象
        samples_per_pixel: 每像素采样数
        render_stokes: 是否渲染 Stokes 参数

    返回:
        dict: {
            'rgb': RGB图像,
            'mask': 掩码,
            'stokes': Stokes参数栈 (如果 render_stokes=True),
            'polarization': 偏振信息字典 (如果 render_stokes=True)
        }
    """
    scene = mi.load_dict(scene_dict)
    # 渲染场景
    image=mi.render(scene, spp=samples_per_pixel)
    # # 获取 film
    # film = sensor.film()
    # bitmap = film.bitmap()
    image_data = np.array(image, dtype=np.float32)

    print(f"  渲染输出形状: {image_data.shape}")
    if render_stokes == True:
        rgba = image_data[..., :4]
        rgb_base = rgba[..., :3]
        alpha = rgba[..., 3]

        s0 = image_data[...,0:3]
        s1 = image_data[...,4:7]
        s2 = image_data[...,8:11]
        full_stokes = np.concatenate([s0,s1,s2],axis=-1)


        result = {}
        result['stokes'] = full_stokes
        result['rgb'] = s0
        # stokes_stack = image_data[..., 4:]
        # result['stokes'] = stokes_stack
        # polarization_info = extract_stokes_parameters(stokes_stack)
        # result['polarization'] = polarization_info
        # rgb = polarization_info['s0']
        # result['rgb'] = rgb

    else:
        normals = image_data[:, :, :3]
        result={}
        
        mask = (normals == 0.).sum(-1) < 3
        normals_vis = (normals+1.0)*0.5
        normals_vis = np.clip(normals_vis,0,1)
        result['normals'] = normals_vis
        result['mask'] = mask

    return result


def generate_multiview_dataset(
        output_dir='./metasurface_dataset',
        n_views=45,
        disk_radius=0.05,
        disk_height=0.02,
        grid_size=4,
        camera_distance=2.5,
        image_resolution=800,
        samples_per_pixel=128,
        cam_fov=45,
        render_stokes=True,  # 🔴 新增: 是否渲染偏振
        save_polarization=False,  # 🔴 新增: 是否保存偏振数据
        no_aovs=False,
):
    """
    生成多视角数据集

    参数:
        output_dir: 输出目录
        n_views: 视角数量
        disk_radius: 圆盘半径
        disk_height: 圆盘高度
        grid_size: 网格大小
        camera_distance: 相机距离
        image_resolution: 图像分辨率
        samples_per_pixel: 每像素采样数
        cam_fov: 相机视场角
        render_stokes: 是否渲染 Stokes 偏振参数
        save_polarization: 是否保存偏振数据(DOP, AOLP等)
        no_aovs: 是否禁用额外输出通道
    """
    # 创建输出目录结构
    output_path = Path(output_dir)
    image_dir = output_path / 'image'
    mask_dir = output_path / 'mask'
    normal_dir = output_path / 'normal'


    if save_polarization:
        polar_dir = output_path / 'polarization'
        dop_dir = polar_dir / 'dop'
        aolp_dir = polar_dir / 'aolp'

        for dir_path in [polar_dir, dop_dir, aolp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    for dir_path in [output_path, image_dir, mask_dir, normal_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f'输出目录: {output_dir}')
    print(f'生成 {n_views} 个视角...')
    print(f'渲染模式: {"Stokes偏振" if render_stokes else "普通RGB"}')

    # 存储相机参数
    camera_dict={}


    # 生成球面均匀采样的视角
    # 使用黄金螺旋采样获得均匀分布
    golden_ratio = (1 + np.sqrt(5)) / 2

    for view_idx in range(n_views):
        # 黄金螺旋采样
        theta = 2 * np.pi * view_idx / golden_ratio  # 方位角
        phi = np.arccos(1 - 2 * (view_idx + 0.5) / n_views)  # 仰角

        # 限制仰角范围 (只从上半球观察)
        phi = np.clip(phi, np.pi / 6, np.pi / 2)  # 30度到90度

        # 生成相机参数
        camera_lookat = lookat_from_spherical(theta, phi, camera_distance)

        print(f'\n视角 {view_idx + 1}/{n_views}')
        print(f'  theta={np.degrees(theta):.1f}°, phi={np.degrees(phi):.1f}°')
        print(f'  相机位置: {camera_lookat[0]}')

        # 创建场景，此处为双通道，创建两次场景，分别提取stokes和法线
        scene_stokes = create_metasurface_scene_dict(
            disk_radius=disk_radius,
            disk_height=disk_height,
            grid_size=grid_size,
            camera_lookat=camera_lookat,
            image_resolution=image_resolution,
            cam_fov=cam_fov,
            render_stokes=True,
            no_aovs=no_aovs,
        )
        result_stokes = render_single_view(scene_stokes, samples_per_pixel,render_stokes=True)


        scene_normals = create_metasurface_scene_dict(
            disk_radius=disk_radius,
            disk_height=disk_height,
            grid_size=grid_size,
            camera_lookat=camera_lookat,
            image_resolution=image_resolution,
            cam_fov=cam_fov,
            render_stokes=False,
            no_aovs=no_aovs,
        )
        result_normals = render_single_view(scene_normals, samples_per_pixel,render_stokes=False)
        result = {}
        result.update(result_stokes)
        result.update(result_normals)
        # 保存文件
        filename = f'{view_idx:04d}'

        # 保存RGB图像 (.exr 格式用于HDR)
        rgb_path = image_dir / f'{filename}.exr'
        # mi.util.write_bitmap(str(rgb_path), result['rgb'])
        mi.util.write_bitmap(str(rgb_path), result['stokes'])
        print(f'  已保存: {rgb_path}')

        # 保存掩码 (.png 格式)
        mask_path = mask_dir / f'{filename}.png'
        # 将掩码转换为3通道图像 (白色=前景, 黑色=背景)
        mask_3ch = np.stack([result['mask']] * 3, axis=-1)
        imageio.imwrite(str(mask_path), (mask_3ch * 255).astype(np.uint8))
        print(f'  已保存: {mask_path}')

        #保存法线
        normal_path = normal_dir / f'{filename}.exr'
        # normal_vis = np.clip((result_normals['normals']+1.0)*0.5,0,1)
        # normal_uint8 = (normal_vis*255).astype(np.uint8)
        mi.util.write_bitmap(str(normal_path), result_normals['normals'])
        #保存为png
        #imageio.imwrite(str(normal_path),normal_uint8)

        # 保存偏振数据
        if save_polarization and 'polarization' in result:
            polar_info = result['polarization']

            # 保存偏振度 (DOP)
            dop_path = dop_dir / f'{filename}.exr'
            mi.util.write_bitmap(str(dop_path), polar_info['dop'])

            # 保存偏振角 (AOLP)
            aolp_path = aolp_dir / f'{filename}.exr'
            mi.util.write_bitmap(str(aolp_path), polar_info['aolp'])

            print(f'  已保存偏振数据: DOP, AOLP')

        # 保存相机参数
        world_mat,scale_mat = lookat_to_world_matrix(
            camera_lookat,
            fov_degrees = cam_fov,
            image_width=image_resolution,
            image_height=image_resolution)

        camera_dict[f'world_mat_{view_idx}'] = world_mat.astype(np.float32)
        camera_dict[f'scale_mat_{view_idx}'] = scale_mat.astype(np.float32)



    # 保存相机参数文件
    camera_file = output_path / 'cameras.npz'
    np.savez(str(camera_file), **camera_dict)
    print(f'\n已保存相机参数: {camera_file}')

    print(f'\n✅ 数据集生成完成!')
    print(f'   总视角数: {n_views}')
    print(f'   输出目录: {output_dir}')
    print(f'   渲染模式: {"Stokes偏振" if render_stokes else "普通RGB"}')
    print(f'\n数据集结构:')
    print(f'  {output_dir}/')
    print(f'  ├── cameras.npz       ({len(camera_dict)} 个相机参数)')
    print(f'  ├── image/            ({n_views} 个 .exr 文件)')
    print(f'  ├── mask/             ({n_views} 个 .png 文件)')
    if save_polarization:
        print(f'  └── polarization/     (偏振数据)')
        print(f'      ├── dop/          ({n_views} 个 .exr 文件)')
        print(f'      └── aolp/         ({n_views} 个 .exr 文件)')


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    generate_multiview_dataset(
        output_dir='./metasurface_dataset_spectral_polarized',
        n_views=45,  # 视角数量
        disk_radius=0.05,  # 圆盘半径
        disk_height=0.08,  # 圆盘高度
        grid_size=3,  # 每个面 4×4 网格
        camera_distance=2.5,  # 相机距离
        image_resolution=800,  # 图像分辨率
        samples_per_pixel=128,  # 采样数
        cam_fov=45,  # 视场角
        render_stokes=True,  # 启用偏振渲染
        save_polarization=False,  # 是否保存偏振数据
    )

    print('\n下一步操作:')
    print('1. 检查生成的数据集文件')
    print('2. 运行验证脚本: python validate_dataset.py ./metasurface_dataset_spectral_polarized')
    print('3. 修改 PANDORA 配置文件指向这个数据集')
    print('4. 运行训练命令:')
    print('   python train.py --conf configs/your_config.yaml')