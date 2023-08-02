import torch.nn as nn

from torchvision.models import vgg16


class ObjectLocalization(nn.Module):
    def __init__(self, output=None):
        super(ObjectLocalization, self).__init__()
        if output is None:
            output = 3

        # VGG16 features
        self.vgg16_model = vgg16()
        self.vgg16_features = self.vgg16_model.features

        # avgpool
        # self.avgpool = self.vgg16_model.avgpool

        # classifier
        self.fc = nn.Sequential(
            nn.Linear(in_features=25088, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=output, bias=True)
        )

    def forward(self, x):
        # x = self.avgpool(self.vgg16_features(x))
        x = self.vgg16_features(x)
        x = x.view(-1, 25088)
        # print(x.size())
        x = self.fc(x)
        return x

    def load_vgg16_weights(self, weights, freeze=False):
        print("Load pretrained vgg16 weights")
        self.vgg16_model.load_state_dict(weights)
        if freeze:
            print("Freeze the vgg16 weights")
            for param in self.vgg16_features.parameters():
                param.requires_grad = False
