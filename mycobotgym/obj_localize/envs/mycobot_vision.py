import os
import numpy as np
import torch
import mujoco
import torchvision.transforms as transforms

from typing import Literal
from os import path
from PIL import Image
from gymnasium_robotics.utils import mujoco_utils
from mycobotgym.envs.mycobot import MyCobotEnv
from mycobotgym.obj_localize.vision_model.model import ObjectLocalization
from mycobotgym.obj_localize.constants import *
from mycobotgym.obj_localize.utils.data_utils import *


class MyCobotVision(MyCobotEnv):
    def __init__(self, model_path: str = MYCOBOT_PATH, has_object=False, block_gripper=False, control_steps=5,
                 controller_type: Literal['mocap', 'IK', 'joint', 'delta_joint'] = 'IK', obj_range: float = 0.1,
                 target_range: float = 0.1, target_offset: float = 0.0, target_in_the_air=True, distance_threshold=0.05,
                 initial_qpos: dict = {}, fetch_env: bool = True, reward_type="sparse", frame_skip: int = 20,
                 default_camera_config: dict = DEFAULT_CAMERA_CONFIG, mode: str = "train", **kwargs) -> None:
        self.mode = mode
        if "eval" == self.mode:
            print(">>>Evaluation.")
            self.vision_model = ObjectLocalization()
            vision_model_pth = path.join(
                path.dirname(path.realpath(__file__)),
                BEST_MODEL_PATH
            )
            self.vision_model.load_state_dict(torch.load(vision_model_pth))

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
            cam_img = self.mujoco_renderer.render("rgb_array", camera_name=BIRD_VIEW)
            # display_img(cam_img)
            with torch.no_grad():
                cam_img = Image.fromarray(cam_img.copy()).resize((224, 224))
                # display_img(cam_img)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                img_tensor = transform(cam_img)
                img_tensor = torch.unsqueeze(img_tensor, dim=0)
                target_pos = self.vision_model(img_tensor)
                target_pos = torch.squeeze(target_pos).cpu().numpy().astype(np.float64)
                # scale to meters
                target_pos = target_pos / 100
            obs_goal["desired_goal"] = target_pos.copy()
        return obs_goal

    def generate_image(self):
        super(MyCobotVision, self).reset_model()
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal
        mujoco.mj_forward(self.model, self.data)
        target_pos = mujoco_utils.get_site_xpos(
            self.model, self.data, "target0")
        cam_img = self.mujoco_renderer.render("rgb_array", camera_name=BIRD_VIEW)
        # display_img(cam_img)
        # print(target_pos)
        # print(self.goal)
        assert target_pos.all() == self.goal.all()
        return target_pos, cam_img
