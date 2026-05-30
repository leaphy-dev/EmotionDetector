import torch
from torch import nn


class EmotionCNNModel1(nn.Module):
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
    NAME = "EmotionCNN Model 1"

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


# class EmotionCNNModel3(nn.Module):
#     """
#         Input: 1x48x48 (灰度图像)
#         Conv1: 32x48x48 (padding=1, 保持尺寸)
#         Conv2: 32x46x46 (无padding) -> MaxPool -> 32x22x22
#         Conv3: 64x20x20 (无padding)
#         Conv4: 64x18x18 (无padding) -> MaxPool -> 64x9x9
#         Conv5: 128x7x7 (无padding)
#         Global Average Pooling: 128x1x1
#         FC: 128 -> 7 (输出7类情感)
#     """
#     NAME = "EmotionCNN Model 3"
#     def __init__(self):
#         super().__init__()
#         # 使用更大的卷积核和不同的架构（类似VGG风格）
#         # 卷积层1: 1x48x48 -> 32x46x46 (padding=1, kernel=3 保持尺寸)
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#
#         # 卷积层2: 32x46x46 -> 32x44x44 (无padding)
#         self.conv2 = nn.Conv2d(32, 32, 3)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(2, 2)  # 44->22
#
#         # 卷积层3: 32x22x22 -> 64x20x20
#         self.conv3 = nn.Conv2d(32, 64, 3)
#         self.bn3 = nn.BatchNorm2d(64)
#
#         # 卷积层4: 64x20x20 -> 64x18x18
#         self.conv4 = nn.Conv2d(64, 64, 3)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(2, 2)  # 18->9
#
#         # 卷积层5: 64x9x9 -> 128x7x7
#         self.conv5 = nn.Conv2d(64, 128, 3)
#         self.bn5 = nn.BatchNorm2d(128)
#
#         # 全局平均池化层（替代全连接层，减少参数量）
#         self.gap = nn.AdaptiveAvgPool2d(1)
#
#         # 全连接层: 128
#         self.fc1 = nn.Linear(128, 7)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         # 第一组卷积（保持尺寸）
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.pool1(x)
#         x = self.dropout(x)
#
#         # 第二组卷积
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.pool2(x)
#         x = self.dropout(x)
#
#         # 第三组卷积
#         x = self.relu(self.bn5(self.conv5(x)))
#
#         # 全局平均池化
#         x = self.gap(x)
#
#         # 展平并分类
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x

class EmotionCNNModel2(nn.Module):
    """
        Input: 1x48x48 (灰度图像)
        Conv1: 32x46x46 (无padding) -> ReLU -> BN
        Conv2: 32x44x44 (无padding) -> ReLU -> BN -> MaxPool -> 32x22x22
        Conv3: 64x20x20 (无padding) -> ReLU -> BN
        Conv4: 64x18x18 (无padding) -> ReLU -> BN -> MaxPool -> 64x9x9
        Conv5: 128x7x7 (无padding) -> ReLU -> BN
        Global Average Pooling: 128x1x1
        FC: 128 -> 7 (输出7类情感)
    """
    NAME = "EmotionCNN Model 2"

    def __init__(self):
        super().__init__()
        # 卷积层1: 1x48x48 -> 32x46x46
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积层2: 32x46x46 -> 32x44x44
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
        self.conv5 =nn.Conv2d(64, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)

        # 全局平均池化层
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc1 = nn.Linear(128, 7)

        self.relu = nn.ReLU()
        # 增加Dropout层，放在池化之后
        self.dropout1 = nn.Dropout(0.5)  # 第一个池化后
        self.dropout2 = nn.Dropout(0.5)  # 第二个池化后
        self.dropout3 = nn.Dropout(0.3)  # GAP后

    def forward(self, x):
        # 第一组卷积
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)  # 加Dropout

        # 第二组卷积
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)  # 加Dropout

        # 第三组卷积
        x = self.relu(self.bn5(self.conv5(x)))

        # 全局平均池化
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout3(x)  # 加Dropout

        x = self.fc1(x)
        return x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    包含通道注意力和空间注意力两个子模块
    """

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()

        # ========== 通道注意力模块 ==========
        # 使用自适应平均池化和最大池化提取全局特征
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的MLP网络
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # ========== 空间注意力模块 ==========
        # 使用7x7卷积学习空间权重
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # 输入x: [B, C, H, W]

        # ----- 通道注意力 -----
        # 平均池化路径
        avg_out = self.shared_mlp(self.avg_pool(x))
        # 最大池化路径
        max_out = self.shared_mlp(self.max_pool(x))
        # 融合并应用sigmoid
        channel_att = torch.sigmoid(avg_out + max_out)

        # 应用通道注意力
        x = x * channel_att

        # ----- 空间注意力 -----
        # 在通道维度上计算平均值和最大值
        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        # 拼接
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)  # [B, 2, H, W]
        # 卷积得到空间权重
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))

        # 应用空间注意力
        x = x * spatial_att

        return x


class EmotionCNNModel3(nn.Module):
    """
        ResNet18 + CBAM注意力机制
        Input: 1x48x48 (灰度图像)
        预训练ResNet18作为backbone
        在每个残差块后插入CBAM模块
        替换分类头为7类表情
    """
    NAME = "EmotionCNN Model 3"

    def __init__(self, pretrained=True):
        super().__init__()

        # 加载预训练ResNet18
        from torchvision import models
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # ----- 修改输入层：1通道转3通道的适配 -----
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                # 将RGB权重平均为单通道
                self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        # ----- 定义CBAM模块，插入在不同位置 -----
        # 每个stage的输出通道数
        # stage1: 64, stage2: 128, stage3: 256, stage4: 512
        self.cbam1 = CBAM(64, reduction=8)
        self.cbam2 = CBAM(128, reduction=8)
        self.cbam3 = CBAM(256, reduction=8)
        self.cbam4 = CBAM(512, reduction=8)

        # ----- 替换分类头 -----
        in_features = int(self.backbone.fc.in_features)  # 512

        # 再替换
        self.backbone.fc = nn.Identity()  # 移除原始分类头

        # 新的分类头（增加dropout防止过拟合）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

        # 保存中间层引用，方便forward
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 初始卷积层
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.relu(x)
        x = self.backbone.maxpool(x)

        # Layer1 (输出64通道) + CBAM
        x = self.backbone.layer1(x)
        x = self.cbam1(x)

        # Layer2 (输出128通道) + CBAM
        x = self.backbone.layer2(x)
        x = self.cbam2(x)

        # Layer3 (输出256通道) + CBAM
        x = self.backbone.layer3(x)
        x = self.cbam3(x)

        # Layer4 (输出512通道) + CBAM
        x = self.backbone.layer4(x)
        x = self.cbam4(x)

        # 全局平均池化
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # 分类
        x = self.classifier(x)

        return x