import mujoco
from mycobotgym.obj_localize.envs.mycobot_vision import MyCobotVision
from mycobotgym.obj_localize.constants import *
from mycobotgym.obj_localize.utils.data_utils import display_img


def temp():
    env = MyCobotVision()
    env.generate_image()


def main():
    env = MyCobotVision()
    target_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "target0"
    )
    env.model.site_pos[target_id] = [
        -0.09130323724538356, -0.15, 0.8877910673378898
    ]
    mujoco.mj_forward(env.model, env.data)
    cam_img = env.mujoco_renderer.render("rgb_array", camera_name=BIRD_VIEW)
    display_img(cam_img)
    cam_img = env.mujoco_renderer.render("rgb_array", camera_name=FRONT_VIEW)
    display_img(cam_img)
    cam_img = env.mujoco_renderer.render("rgb_array", camera_name=SIDE_VIEW)
    display_img(cam_img)


if __name__ == '__main__':
    # temp()
    main()