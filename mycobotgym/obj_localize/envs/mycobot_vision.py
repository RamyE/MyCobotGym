import os
import numpy as np
import torch
import mujoco
import torchvision.transforms as transforms

from typing import Literal
from os import path
from PIL import Image
from mycobotgym.utils import *
from gymnasium_robotics.utils.rotations import quat2euler, euler2quat
from gymnasium_robotics.utils import mujoco_utils
from mycobotgym.envs.mycobot import MyCobotEnv
from mycobotgym.obj_localize.vision_model.model import ObjectLocalization
from mycobotgym.obj_localize.constants import *
from mycobotgym.obj_localize.utils.data_utils import *


MAX_ROTATION_DISPLACEMENT = 0.5


def limit_obj_loc(pos):
    y_threshold = -0.15
    pos[1] = max(pos[1], y_threshold)


class MyCobotVision(MyCobotEnv):
    def __init__(self, model_path: str = MYCOBOT_PATH, has_object=False, block_gripper=False, control_steps=5,
                 controller_type: Literal['mocap', 'IK', 'joint', 'delta_joint'] = 'IK', obj_range: float = 0.1,
                 target_range: float = 0.1, target_offset: float = 0.0, target_in_the_air=True, distance_threshold=0.05,
                 initial_qpos: dict = {}, fetch_env: bool = True, reward_type="sparse", frame_skip: int = 20,
                 default_camera_config: dict = DEFAULT_CAMERA_CONFIG, mode: str = "train", **kwargs) -> None:
        self.mode = mode
        if "eval" == self.mode:
            print("Use vision model for object localization.")
            self.vision_model_xy = ObjectLocalization()
            self.vision_model_z = ObjectLocalization()
            vision_model_pth_xy = path.join(
                path.dirname(path.realpath(__file__)),
                BEST_MODEL_PATH_XY
            )
            vision_model_pth_z = path.join(
                path.dirname(path.realpath(__file__)),
                BEST_MODEL_PATH_Z
            )
            self.vision_model_xy.load_state_dict(torch.load(vision_model_pth_xy))
            self.vision_model_z.load_state_dict(torch.load(vision_model_pth_z))

        super(MyCobotVision, self).__init__(model_path, has_object, block_gripper, control_steps, controller_type, obj_range,
                                            target_range, target_offset, target_in_the_air, distance_threshold, initial_qpos,
                                            fetch_env, reward_type, frame_skip, default_camera_config, **kwargs)

    def generate_mujoco_observations(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = super(MyCobotVision, self).generate_mujoco_observations()
        if self.has_object and "eval" == self.mode:
            # position from vision model
            print(">>>Render.")
            cam_img = self.mujoco_renderer.render("rgb_array", camera_name=BIRD_VIEW)
            cam_img = torch.unsqueeze(cam_img, dim=0)
            object_pos = self.vision_model(cam_img)
            object_pos = torch.squeeze(object_pos).numpy()

        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )

    def _get_obs(self):
        obs_goal = super(MyCobotVision, self)._get_obs()
        if "eval" == self.mode:
            if self.mujoco_renderer.viewer is not None:
                self.mujoco_renderer.viewer._overlays.clear()
            birdview_img = self.mujoco_renderer.render("rgb_array", camera_name=BIRD_VIEW)
            sideview_img = self.mujoco_renderer.render("rgb_array", camera_name=FRONT_VIEW)
            # display_img(cam_img)
            with torch.no_grad():
                birdview_img = Image.fromarray(birdview_img.copy()).resize((224, 224))
                sideview_img = Image.fromarray(sideview_img.copy()).resize((224, 224))
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                birdview_tensor = transform(birdview_img)
                sideview_tensor = transform(sideview_img)
                birdview_tensor = torch.unsqueeze(birdview_tensor, dim=0)
                sideview_tensor = torch.unsqueeze(sideview_tensor, dim=0)
                target_pos = np.empty(3)
                birdview_est = self.vision_model_xy(birdview_tensor)
                sideview_est = self.vision_model_z(sideview_tensor)
                target_pos[:2] = torch.squeeze(birdview_est).cpu().numpy().astype(np.float64)[:2]
                target_pos[-1] = torch.squeeze(sideview_est).cpu().numpy().astype(np.float64)[-1]
                # scale to meters
                target_pos = target_pos / 100
            obs_goal["desired_goal"] = target_pos.copy()
        return obs_goal

    def _rand_position(self):
        position = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3
        )
        position += self.target_offset
        position[2] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            position[2] += self.np_random.uniform(0, 0.2)

        limit_obj_loc(position)
        return position.copy()

    def _rand_gripper(self):
        if self.controller_type == 'IK':
            assert self.controller is not None
            current_eef_pose = self.data.site_xpos[
                self.model_names.site_name2id["EEF"]
            ].copy()
            target_eef_pose = self._rand_position()
            # print(f"[initial] {current_eef_pose}; [target] {target_eef_pose}")
            if self.fetch_env:
                quat_rot = euler2quat(np.zeros(3) * MAX_ROTATION_DISPLACEMENT)
            else:
                quat_rot = euler2quat(sample_euler([[-180, 180], [-180, 180], [-180, 180]]) * MAX_ROTATION_DISPLACEMENT)
            target_orientation = np.empty(4)
            current_eef_quat = np.empty(4)
            mujoco.mju_mat2Quat(current_eef_quat,
                                self.data.site_xmat[self.model_names.site_name2id["EEF"]].copy())
            mujoco.mju_mulQuat(target_orientation,
                               quat_rot, current_eef_quat)

            ctrl_action = np.zeros(7)
            # ctrl_action[-1] = (
            #     self.actuation_center[-1] +
            #     np.random.uniform(-1, 1) * self.actuation_range[-1]
            # )
            distance = goal_distance(target_eef_pose, current_eef_pose)
            itr = 0
            while distance >= self.distance_threshold and itr <500:
                delta_qpos = self.controller.compute_qpos_delta(
                    target_eef_pose, target_orientation
                )
                ctrl_action[:6] = self.data.ctrl.copy()[:6] + delta_qpos[:6]
                self.data.ctrl[:] = ctrl_action
                mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
                new_eef_pose = self.data.site_xpos[
                    self.model_names.site_name2id["EEF"]
                ].copy()
                distance = goal_distance(target_eef_pose, new_eef_pose)
                itr += 1
            # print(f"[current] {new_eef_pose}; Distance: {distance}")
            return target_eef_pose

    def generate_image(self, cam1=BIRD_VIEW, cam2=None):
        self.reset()
        self._rand_gripper()
        target_pos = mujoco_utils.get_site_xpos(
            self.model, self.data, "target0")
        assert target_pos.all() == self.goal.all()
        if cam2 is not None:
            cam_img1 = self.mujoco_renderer.render("rgb_array", camera_name=cam1)
            cam_img2 = self.mujoco_renderer.render("rgb_array", camera_name=cam2)
            return target_pos, cam_img1, cam_img2
        cam_img = self.mujoco_renderer.render("rgb_array", camera_name=cam1)
        # display_img(cam_img)
        # cam_img = self.mujoco_renderer.render("rgb_array", camera_name=SIDE_VIEW)
        # display_img(cam_img)
        return target_pos, cam_img


if __name__ == '__main__':
    print(path.dirname(path.realpath(__file__)))
    print(os.getcwd())