import torch
import torch.nn as nn
from torchsummary import summary


class Model_02(nn.Module):

    def __init__(self, n_input: int = 1, n_classes: int = 3):
        super().__init__()

        # === HYPERPARAMETERS === #
        N_INPUT = n_input
        CONV_1_N_OUT = 32
        CONV_2_N_OUT = 64

        CONV_KERNAL_SIZE = 3
        STRIDE = 1
        PADDING = 2
        POOL_KERNAL_SIZE = 2

        LINEAR_1_IN = 12480
        LINEAR_1_OUT = 64

        # === NETWORK LAYERS === #
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = N_INPUT,
                out_channels = CONV_1_N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride = STRIDE,
                padding = PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = POOL_KERNAL_SIZE)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = CONV_1_N_OUT,
                out_channels = CONV_2_N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride =  STRIDE,
                padding = PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = POOL_KERNAL_SIZE)
        )

        self.dropout = nn.Dropout(p=0.5)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LINEAR_1_IN, LINEAR_1_OUT),
            nn.Linear(LINEAR_1_OUT, n_classes)
        )


    def forward(self, X):
        output = self.dropout(self.conv2(self.conv1(X)))
        logits = self.dense(output)

        return logits
    

    def get_name(self):
        return('model_02')


if __name__ == '__main__':
    model = Model_02()
    summary(model.cuda(), (1, 256, 6))