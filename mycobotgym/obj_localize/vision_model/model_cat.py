import torch
import torch.nn as nn

from mycobotgym.obj_localize.vision_model.model import ObjectLocalization
from torchvision.models import vgg16


class ObjectLocalizationCat(ObjectLocalization):
    def __init__(self, output=None):
        super().__init__()

        if output is None:
            output = 3

        # classifier
        self.fc = nn.Sequential(
            # nn.Linear(in_features=25088 * 2, out_features=4096, bias=True),
            nn.Linear(in_features=25088 * 2, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            # nn.Linear(in_features=4096, out_features=256, bias=True),
            # nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=output, bias=True)
        )

    def forward(self, input1, input2=None):
        assert input2 is not None
        input1 = self.vgg16_features(input1)
        input2 = self.vgg16_features(input2)
        x = torch.cat((input1, input2))
        x = x.view(-1, 25088*2)
        # print(x.size())
        x = self.fc(x)
        return x

