import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                config.input_channels,
                32,
                kernel_size=config.kernel_size,
                stride=config.stride,
                bias=False,
                padding=(config.kernel_size // 2),
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(config.dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, config.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(config.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.logits = nn.Linear(config.features_len * config.final_out_channels, config.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
