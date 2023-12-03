import logging
import os
import torch
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.log import formatter


TestDataset(Dataset):
        def init(self, folder_path):
            self.folder_path = folder_path
            self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            self.file_list = os.listdir(folder_path)

        def len(self):
            return len(self.file_list)

        def getitem(self, idx):
            img_path = os.path.join(self.folder_path, self.file_list[idx])
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            # Extract the ID from the filename
            img_id, _ = os.path.splitext(self.file_list[idx])
            return image, int(img_id)

def test_dataset(model, test_loader, log, result_csv_path):
        model.eval()
        results = {"id": [], "label": []}

        with torch.no_grad():
            for data in test_loader:
                inputs, img_ids = data
                inputs = inputs.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Append results to the dictionary
                results["id"].extend(img_ids.numpy())
                results["label"].extend(predicted.cpu().numpy())

        # Create a new DataFrame for the results
        df = pd.DataFrame(results)

        # Save results to a new CSV file
        df.to_csv(result_csv_path, index=False)
        log.info(f'Test results saved to {result_csv_path}')
