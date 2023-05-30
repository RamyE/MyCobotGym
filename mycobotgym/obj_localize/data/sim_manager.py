import mujoco
import numpy as np
import matplotlib.pyplot as plt
import cv2

from gymnasium_robotics.utils import mujoco_utils
from gymnasium_robotics.utils.rotations import quat2euler, euler2quat
from mycobotgym.obj_localize.utils.data_utils import *

OBJECT_RANGE = 0.2


class SimManager(object):
    def __init__(self, file_path):
        self.model = mujoco.MjModel.from_xml_path(filename=file_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

        mujoco.mj_forward(self.model, self.data)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "birdview")
        self.table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
        self.table2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table2")
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "object0")
        self.light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "light0")

        # store initial values for reset
        self.init_cam_xpos = self.model.cam_pos[self.cam_id].copy()
        self.init_cam_quat = self.model.cam_quat[self.cam_id].copy()
        self.init_cam_fovy = self.model.cam_fovy[self.cam_id].copy()
        self.init_table_color = self.model.geom_rgba[self.table_id-1][:3].copy()
        self.init_table2_color = self.model.geom_rgba[self.table2_id-1][:3].copy()
        self.init_object_color = self.model.site_rgba[self.object_id][:3].copy()
        self.init_light_ambient = self.model.light_ambient[self.light_id].copy()
        self.init_light_diffuse = self.model.light_diffuse[self.light_id].copy()
        self.init_light_spec = self.model.light_specular[self.light_id].copy()

    def _limit_obj_loc(self, pos):
        y_threshold = -0.15
        pos[1] = max(pos[1], y_threshold)

    def _rand_object(self):
        mujoco.mj_forward(self.model, self.data)
        initial_gripper_xpos = mujoco_utils.get_site_xpos(
            self.model, self.data, "EEF"
        ).copy()
        # print(initial_gripper_xpos)
        object_xpos = initial_gripper_xpos
        while np.linalg.norm(object_xpos - initial_gripper_xpos) < 0.2:
            object_xpos = initial_gripper_xpos + np.random.uniform(
                -OBJECT_RANGE, OBJECT_RANGE, size=3
            )
        object_qpos = mujoco_utils.get_joint_qpos(
            self.model, self.data, "object0:joint"
        )
        self._limit_obj_loc(object_xpos)
        assert object_qpos.shape == (7,)
        object_qpos[:3] = object_xpos
        mujoco_utils.set_joint_qpos(
            self.model, self.data, "object0:joint", object_qpos
        )
        mujoco.mj_forward(self.model, self.data)

    def _rand_camera(self):
        # rand cam_pos
        xpos_range = [[-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]
        cam_xpos = self.model.cam_pos[self.cam_id]
        cam_xpos += sample_xyz(xpos_range)

        # rand cam_quat
        # range in degrees
        euler_range = [[-6, 6], [-6, 6], [-6, 6]]
        cam_quat = self.model.cam_quat[self.cam_id].copy()
        cam_quat = quat2euler(cam_quat)
        cam_quat += sample_quat(euler_range)
        self.model.cam_quat[self.cam_id] = euler2quat(cam_quat)

        # rand fovy
        fovy_range = [40, 50]
        self.model.cam_fovy[self.cam_id] = sample_fov(fovy_range)

        mujoco.mj_forward(self.model, self.data)

    def _rand_light(self):
        spec = [np.random.uniform(0.5, 1)] * 3
        diffuse = [np.random.uniform(0.5, 1)] * 3
        ambient = [np.random.uniform(0.5, 1)] * 3
        self.model.light_ambient[self.light_id] = ambient
        self.model.light_diffuse[self.light_id] = diffuse
        self.model.light_specular[self.light_id] = spec

        mujoco.mj_forward(self.model, self.data)

    def _rand_color(self):
        # rand color for 2 tables & object
        color_range = [0, 255]

        self.model.geom_rgba[self.table_id-1][:3] = sample_color(color_range)
        self.model.geom_rgba[self.table2_id-1][:3] = sample_color(color_range)
        self.model.site_rgba[self.object_id][:3] = sample_color(color_range)

        mujoco.mj_forward(self.model, self.data)

    def _reset_model(self):
        self.model.cam_pos[self.cam_id] = np.copy(self.init_cam_xpos)
        self.model.cam_quat[self.cam_id] = np.copy(self.init_cam_quat)
        self.model.cam_fovy[self.cam_id] = np.copy(self.init_cam_fovy)
        self.model.geom_rgba[self.table_id-1][:3] = np.copy(self.init_table_color)
        self.model.geom_rgba[self.table2_id-1][:3] = np.copy(self.init_table2_color)
        self.model.site_rgba[self.object_id][:3] = np.copy(self.init_object_color)
        self.model.light_ambient[self.light_id] = np.copy(self.init_light_ambient)
        self.model.light_diffuse[self.light_id] = np.copy(self.init_light_diffuse)
        self.model.light_specular[self.light_id] = np.copy(self.init_light_spec)

    def get_data(self, cam_name):
        self._reset_model()
        self._rand_object()
        self._rand_camera()
        self._rand_color()
        self._rand_light()
        object_pos = mujoco_utils.get_site_xpos(self.model, self.data, "object0")
        # print(object_pos)
        self.renderer.update_scene(self.data, camera=cam_name)
        cam_img = self.renderer.render()
        # display_img(cam_img)
        return object_pos, cam_img

