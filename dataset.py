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

class DogsVsCatsAugmentor:
    def __init__(self, dataset, num_transforms=2):
        self.dataset = dataset
        self.num_transforms = num_transforms
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ])

    def augment_dataset(self, save_folder=None, augmentation_fraction=0.001):
        augmented_data = []
        total_images = len(self.dataset)
        num_images_to_augment = int(total_images * augmentation_fraction)

        # Randomly select a starting index for the subset
        start_idx = random.randint(0, total_images - 1)

        # Calculate the end index for the subset
        end_idx = min(start_idx + num_images_to_augment, total_images)

        for idx in range(start_idx, end_idx):
            original_image, label = self.dataset[idx]
            label = int(label)

            for i in range(self.num_transforms):
                augmented_image = self.transforms(original_image)
                augmented_data.append((augmented_image, label))

                # Save the augmented image to the specified folder
                if save_folder:
                    label_name = 'cat' if label == 0 else 'dog'
                    augmented_filename = f'augmented_{idx}_{label_name}_{i}.jpg'
                    save_path = os.path.join(save_folder, augmented_filename)
                    torchvision.utils.save_image(augmented_image, save_path)

        return augmented_data

class DogsVsCatsDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.file_list = os.listdir(image_folder)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.file_list[idx])
        image = Image.open(img_name)
        image = self.transforms(image)

        # Extract label from the image name
        if 'dog' in self.file_list[idx]:
            label = 1
        elif 'cat' in self.file_list[idx]:
            label = 0
        return image, label

class DogsVsCatsCompositeDataset(Dataset):
    def __init__(self, original_dataset, augmented_data):
        self.original_dataset = original_dataset
        self.augmented_data = augmented_data

    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_data)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            augmented_idx = idx - len(self.original_dataset)
            return self.augmented_data[augmented_idx]

def denormalize(tensor):
    # Denormalize the image tensor
    denormalized_tensor = tensor.clone()
    denormalized_tensor = denormalized_tensor * 0.5 + 0.5
    denormalized_tensor = denormalized_tensor.clamp(0, 1)
    return denormalized_tensor

def visualize_original_and_augmented(index, augmented_folder, original_dataset):
    # Get the original image and label from the dataset using the provided index
    original_image, original_label = original_dataset[index]

    denormalized_original = denormalize(original_image)
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(F.to_pil_image(denormalized_original))
    plt.title(f"Original\nLabel: {original_label}")
    plt.axis('off')
    plt.show()

    # Find all augmented images in the specified folder with the given index
    augmented_files = [f for f in os.listdir(augmented_folder) if f.startswith(f'augmented_{index}_')]

    for i, augmented_file in enumerate(augmented_files):
        # Load and display each augmented image
        augmented_image = Image.open(os.path.join(augmented_folder, augmented_file))
        plt.subplot(1, len(augmented_files) + 1, i + 2)
        plt.imshow(augmented_image)
        plt.title(f"Aug {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

original_dataset = DogsVsCatsDataset(os.path.join(data_dir, 'train'))

# Create an instance of DogsVsCatsAugmentor
augmentor = DogsVsCatsAugmentor(original_dataset, num_transforms=2)

# Augment the dataset
augmented_data = augmentor.augment_dataset(save_folder='augmented_images')

# Create a composite dataset with both original and augmented data
trainset = DogsVsCatsCompositeDataset(original_dataset, augmented_data)

# visualize_original_and_augmented(index = 23596, augmented_folder = 'augmented_images', original_dataset = composite_dataset)


trainset = DogsVsCatsDataset(os.path.join(data_dir, 'train'))

# prepare loader for test data without augumentation
test_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=test_transform)
