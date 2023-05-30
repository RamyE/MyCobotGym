import os

import torch
import mujoco

from typing import Literal
from os import path
from mycobotgym.envs.mycobot import MyCobotEnv
from mycobotgym.obj_localize.vision_model.model import ObjectLocalization
from mycobotgym.obj_localize.constants import *


class MyCobotVision(MyCobotEnv):
    def __init__(self, model_path: str = MYCOBOT_PATH, has_object=True, block_gripper=False, control_steps=5,
                 controller_type: Literal['mocap', 'IK', 'joint', 'delta_joint'] = 'IK', obj_range: float = 0.1,
                 target_range: float = 0.1, target_offset: float = 0.0, target_in_the_air=True, distance_threshold=0.05,
                 initial_qpos: dict = {}, fetch_env: bool = False, reward_type="sparse", frame_skip: int = 20,
                 default_camera_config: dict = DEFAULT_CAMERA_CONFIG, mode: str = "train", **kwargs) -> None:

        self.vision_model = ObjectLocalization()
        vision_model_pth = path.join(
            path.dirname(path.realpath(__file__)),
            BEST_MODEL_PATH
        )
        self.vision_model.load_state_dict(torch.load(vision_model_pth, map_location=torch.device('cpu')))
        self.mode = mode
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
    
    def reset_model(self):
        # hide target0 by setting the transparency value alpha
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target0")
        self.model.site_rgba[target_id][-1] = 0
        return super(MyCobotVision, self).reset_model()


if __name__ == '__main__':
    print()
