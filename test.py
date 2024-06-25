import os
import torch
import numpy as np
from skimage import color, io
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = io.imread(img_name)
        image = resize(image, (256, 256))
        gray_image = color.rgb2gray(image)
        gray_image = np.expand_dims(gray_image, axis=2)
        if self.transform:
            gray_image = self.transform(gray_image)
        return gray_image, img_name

transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = TestDataset(root_dir='datasets/my_dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = torch.load('checkpoints/your_dataset_pix2pix/latest_net_G.pth')
model.eval()
predictions = []

with torch.no_grad():
    for gray_imgs, img_names in test_loader:
        gray_imgs = gray_imgs.to('cuda')
        outputs = model(gray_imgs)
        outputs = outputs.cpu().numpy()
        predictions.append(outputs)

predictions = np.concatenate(predictions, axis=0)
predictions = (predictions * 255).astype(np.uint8)

# 保存预测结果
np.save("prediction.npy", predictions)
