import torch.nn as nn

from torchvision.models import vgg16


class ObjectLocalization(nn.Module):
    def __init__(self, weights=None, freeze=False):
        super(ObjectLocalization, self).__init__()
        self.vgg16_model = vgg16()
        if weights is not None:
            print("Load pretrained vgg16 weights")
            self.vgg16_model.load_state_dict(weights)

        # VGG16 features
        self.vgg16_features = self.vgg16_model.features
        if freeze:
            assert weights is not None
            print("Freeze the pretrained weights")
            for param in self.vgg16_features.parameters():
                param.requires_grad = False

        # avgpool
        # self.avgpool = self.vgg16_model.avgpool

        # classifier
        self.fc = nn.Sequential(
            nn.Linear(in_features=25088, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=3, bias=True)
            # nn.Linear(in_features=64, out_features=2, bias=True)
        )

    def forward(self, x):
        # x = self.avgpool(self.vgg16_features(x))
        x = self.vgg16_features(x)
        x = x.view(-1, 25088)
        x = self.fc(x)
        return x

