
from collections import OrderedDict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ResNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 3,
    ) -> None:

        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2, bias=True)),
                    #("norm0", nn.BatchNorm3d(num_init_features)),
                    ("norm0", nn.InstanceNorm3d(64)),
                    ("relu0", nn.LeakyReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Block 1
        block1 = ResBlock(n_block='1', in_channels=64, out_channels=64)
        self.features.add_module('res_block_1', block1)

        inter_conv1 = InterConv(n_block='1', in_channels=64, out_channels=128, kernel=3, stride=2, padding=1)
        self.features.add_module('inter_conv_1', inter_conv1)

        # Block 2
        block2 = ResBlock(n_block='2', in_channels=128, out_channels=128)
        self.features.add_module('res_block_2', block2)

        inter_conv2 = InterConv(n_block='2', in_channels=128, out_channels=256, kernel=3, stride=2, padding=1)
        self.features.add_module('inter_conv_2', inter_conv2)

        # Block 3
        block3 = ResBlock(n_block='3', in_channels=256, out_channels=256)
        self.features.add_module('res_block_3', block3)

        inter_conv3 = InterConv(n_block='3', in_channels=256, out_channels=512, kernel=3, stride=2, padding=1)
        self.features.add_module('inter_conv_3', inter_conv3)

        # Block 4
        block4 = ResBlock(n_block='4', in_channels=512, out_channels=512)
        self.features.add_module('res_block_4', block4)

        inter_conv4 = InterConv(n_block='4', in_channels=512, out_channels=512, kernel=1, stride=1, padding=1)
        self.features.add_module('inter_conv_4', inter_conv4)


        self.last_pool = nn.AvgPool3d((512))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        #out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))

        out = th.flatten(out, 1)
        out = self.classifier(out)
        #out = th.softmax(out, dim=1)
        out = th.log_softmax(out, dim=1)
        return out



class ResBlock(nn.Module):
    def __init__(self, n_block, in_channels, out_channels) -> None:
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict([
                (f'conv_{n_block}_1', nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)), ##same padding = 57
                (f'norm_{n_block}_1', nn.InstanceNorm3d(out_channels)),
                (f'relu_{n_block}_1', nn.LeakyReLU()),
                (f'dropout_{n_block}_1', nn.Dropout3d(0.3)),

                (f'conv_{n_block}_2',nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
                (f'norm_{n_block}_2', nn.InstanceNorm3d(out_channels)),
                (f'relu_{n_block}_2', nn.LeakyReLU()),

            ])
        )


    def forward(self, input: Tensor) -> Tensor:
        res = self.features(input)
        out = res + input
        return out


class InterConv(nn.Module):
    def __init__(self, n_block, in_channels, out_channels, kernel, stride, padding) -> None:
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict([
                (f'i_conv_{n_block}', nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)),
                (f'i_norm_{n_block}', nn.InstanceNorm3d(out_channels)),
                (f'i_relu_{n_block}', nn.LeakyReLU())
            ])
        )

    def forward(self, input: Tensor) -> Tensor:
        out = self.features(input)
        return out


# model = ResNet()
#
# ti.summary(model, input_size=(1, 1, 91, 109, 91))


# true = np.array([[0, 1, 0], [0, 0, 1]])
# pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
# weights = class_weight.compute_class_weight(class_weight='balanced', classes=list(config.LABELS.values()),
#                                                 y=true)
#
# print()
#
# weights = th.Tensor(weights)
# true = th.Tensor(true)
# pred = th.Tensor(pred)
#
# pred = th.clamp(pred, 0.0000001, 1 - 0.0000001)
# loss = -1 * (true * th.log(pred))
# print(loss)
#
# loss_final = th.sum(loss) * (1. / pred.shape[0])
# print(loss_final)
#
# pred_nll = th.log_softmax(pred, dim=1)
# loss_nll = nn.NLLLoss()(pred_nll, true)
# print(loss_nll)