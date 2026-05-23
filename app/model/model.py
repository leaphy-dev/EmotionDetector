import torch.nn as nn
class EmotionCNNModel1(nn.Module):
    """
        Input: 1x48x48 (灰度图像)
        Conv1: 32x48x48 -> ReLU -> MaxPool -> 32x24x24
        Conv2: 64x24x24 -> ReLU -> MaxPool -> 64x12x12
        Flatten: 64*12*12 = 9216
        FC1: 9216 -> 128 (ReLU)
        FC2: 128 -> 7 (输出7类情感)
    """
    NAME = "EmotionCNN Model 1"

    def __init__(self):
        super().__init__()
        # 卷积层: 输入1x48x48 -> 32x24x24
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层: 32x24x24 -> 64x12x12
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 48->24
        x = self.pool(self.relu(self.conv2(x)))  # 24->12
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EmotionCNNModel2(nn.Module):
    """
        3个卷积层，每层后接BatchNorm、ReLU和最大池化
        渐进式通道数增长：64 -> 128 -> 256
        3个全连接层，带有Dropout正则化
        BatchNorm加速收敛并提高稳定性

        Input: 1x48x48 (灰度图像)
        Conv1+BN: 64x48x48 -> ReLU -> MaxPool -> 64x24x24
        Conv2+BN: 128x24x24 -> ReLU -> MaxPool -> 128x12x12
        Conv3+BN: 256x12x12 -> ReLU -> MaxPool -> 256x6x6
        Flatten: 256*6*6 = 9216
        FC1: 9216 -> 512 (ReLU + Dropout 0.5)
        FC2: 512 -> 256 (ReLU + Dropout 0.5)
        FC3: 256 -> 7 (输出7类情感)
    """
    NAME = "EmotionCNN Model 2"

    def __init__(self):
        super().__init__()
        # 更深层的卷积网络，增加dropout防止过拟合
        # 卷积层1: 1x48x48 -> 64x24x24
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        # 卷积层2: 64x24x24 -> 128x12x12
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # 卷积层3: 128x12x12 -> 256x6x6
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 全连接层: 256 * 6 * 6 = 9216
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)

        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一层卷积 + 池化
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 48->24
        # 第二层卷积 + 池化
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 24->12
        # 第三层卷积 + 池化
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 12->6
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接层 + dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class EmotionCNNModel3(nn.Module):
    """
        Input: 1x48x48 (灰度图像)
        Conv1: 32x48x48 (padding=1, 保持尺寸)
        Conv2: 32x46x46 (无padding) -> MaxPool -> 32x22x22
        Conv3: 64x20x20 (无padding)
        Conv4: 64x18x18 (无padding) -> MaxPool -> 64x9x9
        Conv5: 128x7x7 (无padding)
        Global Average Pooling: 128x1x1
        FC: 128 -> 7 (输出7类情感)
    """
    NAME = "EmotionCNN Model 3"
    def __init__(self):
        super().__init__()
        # 使用更大的卷积核和不同的架构（类似VGG风格）
        # 卷积层1: 1x48x48 -> 32x46x46 (padding=1, kernel=3 保持尺寸)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积层2: 32x46x46 -> 32x44x44 (无padding)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 44->22

        # 卷积层3: 32x22x22 -> 64x20x20
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)

        # 卷积层4: 64x20x20 -> 64x18x18
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 18->9

        # 卷积层5: 64x9x9 -> 128x7x7
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)

        # 全局平均池化层（替代全连接层，减少参数量）
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 全连接层: 128
        self.fc1 = nn.Linear(128, 7)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 第一组卷积（保持尺寸）
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        # 第二组卷积
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # 第三组卷积
        x = self.relu(self.bn5(self.conv5(x)))

        # 全局平均池化
        x = self.gap(x)

        # 展平并分类
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x