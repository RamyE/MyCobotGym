import torch
import torch.nn as nn

from mycobotgym.obj_localize.vision_model.model import ObjectLocalization
from torchvision.models import vgg16


class ObjectLocalizationCat(ObjectLocalization):
    def __init__(self, output=None):
        super().__init__()

        if output is None:
            output = 3

        # decision network
        self.decision = nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
        )
        # classifier
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=output, bias=True)
        )

    def forward(self, input1, input2=None):
        assert input2 is not None
        input1 = self.vgg16_features(input1)
        input2 = self.vgg16_features(input2)
        x = torch.cat((input1, input2), dim=1)
        x = self.decision(x)
        x = x.view(-1, 64*7*7)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    vgg16 = vgg16()
    # backbone = torch.nn.Sequential(*list(vgg16.features.children()))
    backbone = vgg16.features

    left_image = torch.randn(1, 3, 224, 224)  # Example input shape
    right_image = torch.randn(1, 3, 224, 224)  # Example input shape

    # Extract features from both left and right images
    left_features = backbone(left_image)
    right_features = backbone(right_image)

    # print(left_features.size())

    # Perform feature fusion (e.g., concatenation)
    combined_features = torch.cat((left_features, right_features), dim=1)

    print(combined_features.size())

    # output = localization_head(combined_features)
    #
    # print(output.size())
    # print(backbone)
