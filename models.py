"""
This file contains three models needed to build ADDA:
    - Encoder: to extract features.
    - Classifier: to perform classification.

SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation (ICCV 2021)
"""

from torch import nn
from utils import GradientReversal


class Encoder(nn.Module):
    """
    encoder for SENTRY.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

        # self.layer = nn.Sequential(
        #     nn.Linear(2, 20),
        #     nn.ReLU(),
        #     nn.Linear(20, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 2)
        # )

    def forward(self, input):
        out = self.layer(input)
        return out


class Classifier(nn.Module):
    """
    classifier for SENTRY.
    """
    def __init__(self):
        super(Classifier, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.LogSoftmax()
        )

    def forward(self, input):
        out = self.layer(input)
        return out
