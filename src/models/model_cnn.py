from torch import nn



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1= nn.BatchNorm2d(16)
        self.pool1= nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2= nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn2= nn.BatchNorm2d(64)
        self.pool2= nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3= nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3= nn.BatchNorm2d(128)
        self.pool3= nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1= nn.Linear(128*3*3, 29)


    def forward(self, x):
        out = self.conv1(x)
        out=self.bn1(out)
        out = nn.ReLU()(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = nn.ReLU()(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc1(out)  # Fully connected layer

        return out