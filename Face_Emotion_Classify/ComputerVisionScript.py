import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import shutil
import urllib
import os
import random
from sklearn.model_selection import train_test_split
import zipfile

import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataHandler:
    def __init__(self):
        self.default_transform = transforms.Compose([transforms.ToTensor()])
        
    def download_data(self, url, extract_path="data", save_path="data.zip"):
        urllib.request.urlretrieve(url, save_path)
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
                
    def delete_folder(self, folder_path):
        shutil.rmtree(folder_path)
        
    def split_data_folder(self, data_dir, train_dir, test_dir, test_size=0.2):
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)

            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)

            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            image_paths = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]
            train_paths, test_paths = train_test_split(image_paths, test_size=test_size, random_state=42)

            for path in train_paths:
                filename = os.path.basename(path)
                dest_path = os.path.join(train_class_dir, filename)
                shutil.copy2(path, dest_path)

            for path in test_paths:
                filename = os.path.basename(path)
                dest_path = os.path.join(test_class_dir, filename)
                shutil.copy2(path, dest_path)
                
    def data_augment(self, train_dir, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2):
        class_names = os.listdir(train_dir)
        num_images = {class_name: len(os.listdir(os.path.join(train_dir, class_name))) for class_name in class_names}
        print(f'Number of images in each class: {num_images}')

        target_num = max(num_images.values())

        data_transform = transforms.Compose([
            transforms.RandomRotation(rotation_range),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.RandomAffine(degrees=0, translate=(width_shift_range, height_shift_range),
                                    shear=shear_range, scale=(1 - zoom_range, 1 + zoom_range)),
            transforms.ToTensor()
        ])

        for class_name in class_names:
            num_to_generate = target_num - num_images[class_name]
            if num_to_generate <= 0:
                continue
            print(f'Number of Images needed to be generated for class {class_name}: {num_to_generate}')

            class_path = os.path.join(train_dir, class_name)
            print(f'Images need to be generated in: {class_path}')

            if not os.path.exists(train_dir):
                print(f'Error: Root directory "{train_dir}" does not exist.')
                continue

            dataset = ImageFolder(train_dir, transform=data_transform)
            class_indices = dataset.class_to_idx
            class_index = class_indices[class_name]

            class_images = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == class_index]
            class_images = random.choices(class_images, k=num_to_generate)

            for i, image in enumerate(class_images):
                new_img_name = f'{class_name}_{num_images[class_name] + i}.jpg'
                new_img_path = os.path.join(class_path, new_img_name)
                torchvision.utils.save_image(image, new_img_path)

            num_images[class_name] += num_to_generate

        print(f'Number of images in each class after data augmentation: {num_images}')

  
    def calculate_mean_std(self, data_dir):
        dataset = ImageFolder(data_dir, transform=self.default_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        mean = 0.0
        std = 0.0
        total_samples = 0

        for images, _ in data_loader:
            batch_size = images.size(0)
            images = images.view(batch_size, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_size

        mean /= total_samples
        std /= total_samples

        return mean, std
    
    def make_dataset(self, image_dir, transform=None):
        if transform is None:
            transform = self.default_transform
            
        dataset = ImageFolder(data_dir, transform=transform)
        
        return dataset
    
    def make_preloaded_dataset(self, preloaded_data, transform=None, *args, **kwargs):
        """
        Example to use:

        from torchvision.datasets import CIFAR10

        trainloader = load_preloaded_data(CIFAR10, root='./data',
                                          download=True, transform=transform,
                                          train=True, batch_size=16, shuffle=True)

        testloader = load_preloaded_data(CIFAR10, root='./data',
                                         download=True, transform=transform,
                                         train=False, batch_size=16, shuffle=True)
        """
        if transform is None:
            transform = self.default_transform

        dataset = preloaded_data(*args, **kwargs, transform=transform)
        

        return dataset

    
    def images_classes_counter(self, dataset, classes):
        class_counts = {}
        for image, label in dataset:
            label = classes[label]
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        for class_label, count in class_counts.items():
            print(f"Class {class_label}: {count} images")
            
        class_counts = {}
            
        
    def train_test_split(self, train_dataset, split_ratio=0.2):
        num_samples = len(train_dataset)
        num_test = int(num_samples * split_ratio)
        num_train = int(num_samples) - num_test
        
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_test])
        
        return train_dataset, test_dataset
        
    def load_data(self, dataset, batch_size, shuffle=True):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
            
    def image_plotter(self, dataset, class_names, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), num_images=32, figsize=(10, 5), nrows=4, ncols=8):
        random_indices = torch.randperm(len(dataset))[:num_images]
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5))
        for i, index in enumerate(random_indices):
            row = i // ncols
            col = i % ncols
            image, label = dataset[index]  
            ax = axes[row, col]  
            image = image * np.expand_dims(std, axis=(1, 2)) + np.expand_dims(mean, axis=(1, 2))
            image = np.transpose(image, (1, 2, 0))
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(class_names[label])

        plt.tight_layout()
        plt.show()
        
        
        
