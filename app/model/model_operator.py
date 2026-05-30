# model_operator.py
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from app.model.model import EmotionCNNModel1, EmotionCNNModel2, EmotionCNNModel3


class ModelOperator:
    def __init__(self):
        self._logger = logging.Logger(name="ModelOperator")

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not hasattr(torch._C, "_cuda_getDeviceCount"):
            self._logger.warning("Torch not compiled with CUDA enabled")

        self._logger.info(f'使用设备: {self.device}')

        if torch.cuda.is_available():
            self._logger.info(str(subprocess.run("nvidia-smi -L", capture_output=True, text=True).stdout))

    def load_data(self, batch_size=64, use_augmentation=False):
        if use_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_data = datasets.FER2013(root='./dataset', split='train', transform=train_transform)
        test_data = datasets.FER2013(root='./dataset', split='test', transform=test_transform)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        self._logger.info(f'训练集大小: {len(train_data)}')
        self._logger.info(f'测试集大小: {len(test_data)}')

        return train_loader, test_loader

    def train_model(self,
                    model_class,
                    model_path: str = "./model_data/emotion_model.pth",
                    epochs=5,
                    force: bool = False,
                    process_callback=None,
                    use_augmentation=True,
                    use_class_weights=False):

        model_path: Path = Path(model_path)

        if model_path.exists() and not force:
            self._logger.info(f"模型文件{model_path}存在，跳过")
            return None

        name = getattr(model_class, "NAME", "")
        self._logger.info(f'开始训练: {name}...')

        model_instance = model_class().to(self.device)

        if use_class_weights:
            class_counts = [4983, 547, 5121, 8989, 6077, 4002, 6198]
            class_weights = torch.tensor([1.0 / c for c in class_counts])
            class_weights = class_weights / class_weights.sum() * 7
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            self._logger.info(f'使用类别权重训练')
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model_instance.parameters(), lr=0.001)
        train_loader, test_loader = self.load_data(use_augmentation=use_augmentation)

        best_test_acc = 0
        best_model_state = None  # 存内存里

        for epoch in range(epochs):
            model_instance.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model_instance(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = 100. * correct / total

            # 测试
            model_instance.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model_instance(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_acc = 100. * correct / total

            if process_callback:
                process_callback(int(((epoch + 1) / epochs) * 100))

            self._logger.info(f'{name}: Epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(train_loader):.4f} | '
                              f'Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # 深拷贝模型参数到内存
                best_model_state = {k: v.cpu().clone() for k, v in model_instance.state_dict().items()}
                self._logger.info(f'{name}:更新最佳模型，ACC: {test_acc:.2f}%')

        if best_model_state is not None:
            if not model_path.parent.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_model_state, model_path)
            self._logger.info(f'{name}:训练完成，保存最佳模型到 {model_path}，ACC: {best_test_acc:.2f}%')
        else:
            # 没有更好的模型保存最后一个
            torch.save(model_instance.state_dict(), model_path)
            self._logger.info(f'{name}:训练完成，保存最终模型到 {model_path}')

        return model_instance


    def load_model(self, model_class, model_path='emotion_model.pth'):
        if os.path.exists(model_path):
            model_instance = model_class().to(self.device)
            model_instance.load_state_dict(torch.load(model_path, map_location=self.device))
            model_instance.eval()
            self._logger.info(f'加载已训练模型: "{model_path}"...')
            return model_instance
        else:
            self._logger.info(f'未找到模型文件 {model_path}')
            return None

    def predict_image(self, model_instance, image) -> Dict:
        """
        预测图片中的情感
        image: PIL Image 或 numpy array
        model_data: 训练好的模型

        返回: dict, 键为情感索引(0-6), 值为对应的概率
        """
        # 处理图片
        if isinstance(image, np.ndarray):
            # 如果是大图片，先缩小用于显示
            original_height, original_width = image.shape[:2]
            max_display_size = 800  # 最大显示尺寸

            if original_height > max_display_size or original_width > max_display_size:
                scale = max_display_size / max(original_height, original_width)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                display_image = cv2.resize(image, (new_width, new_height))
                self._logger.info(f"图片已缩放: {original_width}x{original_height} -> {new_width}x{new_height}")
            else:
                display_image = image.copy()

            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        else:
            display_image_rgb = np.array(image)

        # 转为灰度图并调整大小（用于模型输入）
        gray = image.convert('L')
        gray = gray.resize((48, 48))

        model_instance.eval()

        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        tensor = transform(gray).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = model_instance(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, idx = torch.max(probs, 1)

        emotions = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '中性']
        prob_dict = {}

        # 打印所有情感的概率
        self._logger.info("=" * 50)
        self._logger.info("情感识别结果:")
        for i, prob in enumerate(probs[0].cpu().numpy()):
            prob_dict[i] = float(prob)
            self._logger.info(f"  {i}:{emotions[i]}: {prob:.2%}")

        confidence_score = confidence.item()
        self._logger.info(f"-" * 50)
        self._logger.info(f"预测结果: {emotions[idx.item()]} (索引{idx.item()}), 置信度: {confidence_score:.2%}")
        self._logger.info("=" * 50)

        # img_with_box = img_for_display.copy()
        # img_with_box = cv2.cvtColor(img_with_box, cv2.COLOR_RGB2BGR)
        #
        # label = f'{emotions[idx.item()]}: {confidence_score:.2%}'
        # cv2.putText(img_with_box, label, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return prob_dict


    def predict_webcam(self,model, image):
        """处理摄像头输入"""
        if image is None:
            return None

        result_img, result_text = self.predict_image(model, image)
        return result_img, result_text

if __name__ == "__main__":
    model_op = ModelOperator()
    model_op.train_model(epochs=5, model_class=EmotionCNNModel1, model_path="../../model_data/EmotionCNNModel1.pth")
    model_op.train_model(epochs=5, model_class=EmotionCNNModel2, model_path="../../model_data/EmotionCNNModel2.pth")
    model_op.train_model(epochs=5, model_class=EmotionCNNModel3, model_path="../../model_data/EmotionCNNModel3.pth")