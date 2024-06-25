import os
import numpy as np
from skimage import color, io
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if
                            os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(
                                ('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # 调试信息
        print(f"Found {len(self.image_files)} image files in {root_dir}")
        for img_file in self.image_files:
            print(img_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = io.imread(img_name)
        image = resize(image, (256, 256))
        gray_image = color.rgb2gray(image)
        gray_image = np.expand_dims(gray_image, axis=2)
        if self.transform:
            image = self.transform(image).float()  # 转换为 float 类型
            gray_image = self.transform(gray_image).float()  # 转换为 float 类型
        return {'A': gray_image, 'B': image, 'A_paths': img_name, 'B_paths': img_name}


transform = transforms.Compose([
    transforms.ToTensor(),
])


def get_train_loader(root_dir, batch_size=16):
    train_dataset = ColorizationDataset(root_dir=root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
