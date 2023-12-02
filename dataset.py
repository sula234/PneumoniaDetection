import torch
import torchvision
from os import path
from pathlib import Path


# define transform functions which should applied to images
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# prepare loader for train data
data_dir = Path('data')
trainset = torchvision.datasets.ImageFolder(root=path.join(data_dir, 'train'), transform=transform )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

# prepare loader for test data
testset = torchvision.datasets.ImageFolder(root=path.join(data_dir, 'val'), transform=transform)

# save all classes to list
classes = ('cat', 'dog')