import pandas as pd
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as tx
from helper import *
import torch
import torchvision.transforms as T

to_gray = config['pre'].getboolean("to_gray")
rotate = config['pre'].getboolean("rotate")
gussian_blur = config['pre'].getboolean("GaussianBlur")
color_jitter = config['pre'].getboolean("ColorJitter")
sharpness = config['pre'].getboolean("sharpness_adjuster")
color_equalizer = config['pre'].getboolean("equalizer")
def get_dataset(name, augment):
    class NailDataset():
        def __init__(self, annotations_file, augment=False, img_dir=None, transform=None, target_transform=None):
            self.df = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform
            self.augment = augment

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            #gray.shape = (256,256)
            image_path = self.df["image_path"][idx]
            mask_path = self.df["mask_path"][idx]
            
            image = cv2.imread(image_path)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(int)
            origin_image = image[:,:,::-1]
            image = image[:,:,::-1]
            image = image[:,:,::-1]
            if to_gray:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            if self.augment:
                image, mask = augmentation(image, mask)                                       
            image = tx.to_tensor(image)
            mask = tx.to_tensor(mask)          
            if to_gray:
                image = tx.normalize(image, [0.5], [0.5]) 
            else:
                image = tx.normalize(image, [0.5,0.5,0.5], [0.5,0.5,0.5]) 
            return image.clone(), mask.clone(), origin_image.copy()
    def augmentation(image, mask):
        #color jitter
            if(random.random()>0.5 and color_jitter):
                jitter = T.ColorJitter(brightness=.5, hue=.3)
                image = jitter(image)
                
                
                
            #Gaussian blur
            if(random.random()>0.5 and gussian_blur):
                blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
                image = blurrer(image)
                mask = blurrer(mask)
                
                
            #Perspective
#             if(random.random()>0.5):
#                 perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
#                 image = perspective_transformer(image)
                
                
            #Rotation
            if(random.random()>0.5 and rotate):
                random_degree = random.randint(0, 8) * 45
                image = tx.rotate(image, random_degree)
                mask = tx.rotate(mask, random_degree)
                
                
                
            #Affine
#             if(random.random()>0.5):
#                 affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
#                 image = affine_transfomer(image)
               
            
            #ElasticTransform
#             if(random.random()>0.5):
#                 elastic_transformer = T.ElasticTransform(alpha=250.0)
#                 image = elastic_transformer(image)
                
            #Invert
#             if(random.random()>0.5):
#                 inverter = T.RandomInvert()
#                 image = inverter(image)
           
            #Posterize
#             if(random.random()>0.5):
#                 posterizer = T.RandomPosterize(bits=2)
#                 image = posterizer(image)
#                 mask = posterizer(mask)
                
                
            #Solarize
#             if(random.random()>0.5):
#                 solarizer = T.RandomSolarize(threshold=192.0)
#                 image = solarizer(image)
                
                
            #AdjustSharpness
#             if(random.random()>0.5 and sharpness):
#                 sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=1)
#                 image = sharpness_adjuster(image)
#                 mask = sharpness_adjuster(mask)
                
                
            #Equalize
            if(random.random()>0.5 and color_equalizer):
                equalizer = T.RandomEqualize()
                image = equalizer(image)
            

            return image, mask
        
        
    if name == "train":
        return NailDataset("train_data.csv", augment)
    elif name == "val":
        return NailDataset("val_data.csv", augment)
        
#data = NailDataset("data.csv")
# dataloader = DataLoader(data, batch_size=1, shuffle=True)
# return dataloader