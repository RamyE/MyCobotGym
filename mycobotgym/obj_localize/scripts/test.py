import os
import torch
import mujoco
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from mycobotgym.obj_localize.constants import *
from mycobotgym.obj_localize.envs.mycobot_vision import MyCobotVision
from mycobotgym.obj_localize.vision_model.model import ObjectLocalization
from mycobotgym.obj_localize.vision_model.model_cat import ObjectLocalizationCat

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model():
    model1 = ObjectLocalizationCat()
    model2 = ObjectLocalization()
    env = MyCobotVision()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    target_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "target0"
    )
    target_pos = [0.02745002, -0.11349673,  0.80998082]
    env.model.site_pos[target_id] = target_pos
    mujoco.mj_forward(env.model, env.data)

    birdview_img = env.mujoco_renderer.render("rgb_array", camera_name=BIRD_VIEW)
    sideview_img = env.mujoco_renderer.render("rgb_array", camera_name=FRONT_VIEW)
    birdview_img = Image.fromarray(birdview_img.copy()).resize((224, 224))
    sideview_img = Image.fromarray(sideview_img.copy()).resize((224, 224))
    birdview_tensor = transform(birdview_img)
    sideview_tensor = transform(sideview_img)
    birdview_tensor = torch.unsqueeze(birdview_tensor, dim=0)
    sideview_tensor = torch.unsqueeze(sideview_tensor, dim=0)

    outpout1 = model1(birdview_tensor, sideview_tensor)
    outpuot2 = model2(birdview_tensor)


def temp():
    model = ObjectLocalization().to(DEVICE)
    weights_path = '../vision_model/trained_models/reach_target/VGGNet_epoch49.pth'
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    env = MyCobotVision()
    target_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "target0"
    )
    target_pos = [0.02745002, -0.11349673,  0.80998082]
    env.model.site_pos[target_id] = target_pos
    mujoco.mj_forward(env.model, env.data)
    cam_img = env.mujoco_renderer.render("rgb_array", camera_name=BIRD_VIEW)
    cam_img = Image.fromarray(cam_img).resize((224, 224))
    image_tensor = transform(cam_img)

    model.eval()
    with torch.no_grad():
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        output = model(image_tensor.to(DEVICE))
        output = torch.squeeze(output).cpu().numpy()
        output = output / 100
    print(f"Real: {target_pos}; Estimate: {output}; Error: {np.linalg.norm(target_pos-output, axis=-1)}")


def main():
    env = MyCobotVision(mode='eval')
    success = 0
    num_itr = 1
    for _ in tqdm(range(num_itr)):
        obs = env.reset_model()
        real = env.goal.copy()
        est = obs.get('desired_goal')
        err = np.linalg.norm(est - real, axis=-1)
        if err < 0.05:
            success += 1
        print(f"Target: {real}")
        print(f"Estimate: {est}")
    print(f"Success rate: {success / num_itr}")


if __name__ == '__main__':
    # main()
    # temp()
    test_model()
    # pos1 = np.array([0.02745002, -0.11349673,  0.80998082])
    # pos2 = np.array([ 0.02142225, -0.11864639,  0.7976401 ])
    # print(np.linalg.norm(pos1 - pos2, axis=-1))