from pathlib import Path
import logging
import os
import torch
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.log import formatted
from test.py import TestDataset
from test.py import test_dataset

# prepare loader for test data
data_dir = Path('data')

model_name_to_test = "PretrainedVGG16"  
test_model = get_model_from_cfg(model_name_to_test)
test_model.load_state_dict(torch.load("/content/ROBT407Final/models/Exp_0/PretrainedVGG16.pth"))  
test_model.cuda()

# Define dataloader for testing with the test dataset
test_dataset = TestDataset(folder_path="data/test1", transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

# Specify the path to save the new CSV file
result_csv_path = "results.csv"  # Replace with the desired path and file name

 # Test the model on the custom test dataset and save results to a new CSV file
test_dataset(test_model, testloader, log, result_csv_path)
