import torch
import torchvision
from torchvision.transforms import ToTensor


# define transform functions which should applied to images
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# prepare loader for train data
trainset = torchvision.datasets.ImageFolder(root='./petfaces/train', transform=transform )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

# prepare loader for test data
testset = torchvision.datasets.ImageFolder(root='./petfaces/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

# save all classes to list
classes = ('cat', 'dog')