import pandas as pd
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
def get_dataset():
    class NailDataset():
        def __init__(self, annotations_file, img_dir=None, transform=None, target_transform=None):
            self.df = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            image_path = self.df["image_path"][idx]
            mask_path = self.df["mask_path"][idx]

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[:,:,::-1]
#             gray = cv2.equalizeHist(gray)
            for i in range(256):
                for j in range(256):
                    if(gray[i][j]>0):
                        gray[i][j] = np.log(gray[i][j])
#             gray = np.log(gray)
            gray = gray.reshape(1,256,256)
#             image = np.moveaxis(image, (0,1,2), (1,2,0))
            
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray.copy(), mask.copy(), image.copy()
    return NailDataset("data.csv")
#data = NailDataset("data.csv")
# dataloader = DataLoader(data, batch_size=1, shuffle=True)
# return dataloader