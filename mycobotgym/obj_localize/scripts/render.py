import mujoco
from mycobotgym.obj_localize.envs.mycobot_vision import MyCobotVision
from mycobotgym.obj_localize.constants import *
from mycobotgym.obj_localize.utils.data_utils import display_img

if __name__ == '__main__':
    env = MyCobotVision()
    target_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "target0"
    )
    env.model.site_pos[target_id] = [
        -0.15005322, -0.14015559,  0.80998082
    ]
    mujoco.mj_forward(env.model, env.data)
    cam_img = env.mujoco_renderer.render("rgb_array", camera_name=BIRD_VIEW)
    display_img(cam_img)
    cam_img = env.mujoco_renderer.render("rgb_array", camera_name=FRONT_VIEW)
    display_img(cam_img)