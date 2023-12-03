from pathlib import Path

# prepare loader for test data
data_dir = Path('data')

model_name_to_test = "PretrainedVGG16"  
test_model = get_model_from_cfg(model_name_to_test)
test_model.load_state_dict(torch.load("pretrainedvgg16_checkpoint.pth"))  
test_model.cuda()

# Define dataloader for testing with the test dataset
test_dataset = TestDataset(folder_path="path/to/your/test_folder", transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

# Specify the path to save the new CSV file
result_csv_path = "results.csv"  # Replace with the desired path and file name

 # Test the model on the custom test dataset and save results to a new CSV file
test_dataset(test_model, testloader, log, result_csv_path)
