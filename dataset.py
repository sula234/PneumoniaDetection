from pathlib import Path

# prepare loader for train data
data_dir = Path('data')

from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import random

class DogsVsCatsDataset(Dataset):
    def __init__(self, image_folder, augment_prob=0.4):
        self.image_folder = image_folder
        self.augment_prob = augment_prob

        self.default_transforms = [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.augumentation = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            ]
        
        self.file_list = self._build_file_list()
        self.dataset_size = len(self.file_list)


    def __len__(self):
        return self.dataset_size
    

    def _build_file_list(self):
        file_list = []
        for class_name in os.listdir(self.image_folder):
            class_path = os.path.join(self.image_folder, class_name)
            file_list.extend([(class_name, file) for file in os.listdir(class_path)])
           
        return file_list

    def __getitem__(self, idx):
        
        class_name, img_name = self.file_list[idx]
        img_path = os.path.join(self.image_folder, class_name, img_name)
        original_image = Image.open(img_path).convert('RGB')  # Adjust the mode based on your images


        #apply augumentation with given probabilty 
        if random.random() <= self.augment_prob:
            new_transforms = self.default_transforms + self.augumentation
            t = transforms.Compose(new_transforms)
        else:
            t = transforms.Compose(self.default_transforms)
            
        original_image = t(original_image)
        class_label = 0 if class_name == 'cats' else 1  # Assuming 'cat' is class 0 and 'dog' is class 1

        
        return original_image, class_label

    

# def visualizeAugmentedImage(dataset, index_to_visualize=0):
#   original_image, original_label = dataset[index_to_visualize]
#   augmented_images, augmented_label = dataset[index_to_visualize]
#   # Create a figure to display images
#   num_images = len(augmented_images) + 1
#   fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
#   axes = axes.flatten()

#   # Visualize original image
#   axes[0].imshow(F.to_pil_image(original_image))
#   axes[0].set_title(f"Original\nLabel: {original_label}")
#   axes[0].axis('off')

#   # Visualize augmented images
#   for i in range(1, num_images):
#     axes[i].imshow(F.to_pil_image(augmented_images[i - 1]))
#     axes[i].set_title(f"Augmented\nLabel: {augmented_label}")
#     axes[i].axis('off')

#   plt.tight_layout()
#   plt.show()


trainset = DogsVsCatsDataset(os.path.join(data_dir, 'train'))

# prepare loader for test data without augumentation
test_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=test_transform)