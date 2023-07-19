import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from mycobotgym.obj_localize.envs.mycobot_vision import MyCobotVision
from mycobotgym.obj_localize.vision_model.model import ObjectLocalization

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def temp():
    env = MyCobotVision(mode='eval')
    model = ObjectLocalization().to(DEVICE)
    weights_path = '../vision_model/trained_models/reach_target/VGGNet_epoch49.pth'
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    obs = env.reset_model()
    print(env.goal)
    print(obs.get('desired_goal'))

    image = env.mujoco_renderer.render("rgb_array", camera_name='birdview')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.fromarray(image).resize((224, 224))
    image_tensor = transform(image)
    model.eval()
    with torch.no_grad():
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        output = model(image_tensor.to(DEVICE))
        output = torch.squeeze(output).cpu().numpy()
    print(output / 100)

def main():
    env = MyCobotVision(mode='eval')
    success = 0
    num_itr = 500
    for i in tqdm(range(num_itr)):
        obs = env.reset_model()
        real = env.goal.copy()
        est = obs.get('desired_goal')
        err = np.linalg.norm(est - real, axis=-1)
        is_success = env._is_success(est, real)
        if err < 0.05:
            assert is_success == True
            success += 1
        # print(f"Target: {real}")
        # print(f"Estimate: {est}")
    print(f"Success rate: {success / num_itr}")



if __name__ == '__main__':
    main()
    # temp()