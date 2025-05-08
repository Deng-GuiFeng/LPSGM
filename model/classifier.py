import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, feat_dim, num_classes=5):
        super(Classifier, self).__init__()
        self.ln = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.ln(x)


if __name__ == "__main__":
    model = Classifier()
    print(sum(p.numel() for p in model.parameters()))

