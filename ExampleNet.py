import torch.nn as nn


class ExampleNet(nn.Module):
    def __init__(
            self,
            out_channels,
    ):
        super(ExampleNet, self).__init__()

        layers = []
        layers.append(
            nn.Conv3d(
                in_channels=1,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv3d(
                in_channels=10,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        self.add_module('layers', nn.Sequential(*layers))
        self.init_weights()

    def init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.xavier_uniform_(m.bias.data)

    def forward(
            self,
            img,
    ):
        for i, layer in enumerate(self.layers):
            img = layer(img)

        return img





