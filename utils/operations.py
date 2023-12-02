import torch
from tqdm import tqdm
from model import CNN, MLP
import os

EXPs_PATH = "models"

def validate(net,test_loader,device):
  count=acc=0
  for xo,yo in tqdm(test_loader,desc="Validation"):
    x,y = xo.to(device), yo.to(device)
    with torch.no_grad():
      p = net.forward(x)
      _,predicted = torch.max(p,1)
      acc+=(predicted==y).sum()
      count+=len(x)
  return acc/count

def train(net,train_loader,test_loader,epochs,loss_fn, optimizer, log, model_name):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  log.info(f'Using device={device}') 
  net = net.to(device)

  for ep in range(epochs):
    count=acc=0
    running_loss = 0.0
    for xo,yo in tqdm(train_loader, desc="Training"):
      x = xo.to(device)
      y = yo.to(device)
      optimizer.zero_grad()
      p = net.forward(x)
      loss = loss_fn(p,y)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      _,predicted = torch.max(p,1)
      acc+=(predicted==y).sum()
      count+=len(x)
    val_acc = validate(net,test_loader,device=device)
    log.info(f"Epoch={ep}, Train accuracy={acc/count}, Validation accuracy={val_acc}, Loss={running_loss/len(train_loader)}")

  # save model
  checkpoint = {
    "model": net.state_dict(),
    "optimizer": optimizer.state_dict()
    }
  return checkpoint


def get_model_from_cfg(model: str):
  
  if model.upper() == "CNN":
    return CNN()
  
  elif model.upper() == "MLP":
    return MLP()
  
def create_new_experiment():
  # Check existing folders in the models directory
    existing_folders = [folder for folder in os.listdir(EXPs_PATH) if folder.startswith("Exp_")]

    if not existing_folders:
        # If no existing folders, create "Exp_0"
        new_exp_folder = os.path.join(EXPs_PATH, "Exp_0")
    else:
        # Find the next available ID
        existing_ids = [int(folder.split("_")[1]) for folder in existing_folders]
        next_id = max(existing_ids) + 1 if existing_ids else 0

        # Create the new folder
        new_exp_folder = os.path.join(EXPs_PATH, f"Exp_{next_id}")

    # Create the directory if it doesn't exist
    os.makedirs(new_exp_folder, exist_ok=True)
    return new_exp_folder


def save_experiment(new_exp_folder, model_names, checkpoints):
  
  for checkpoint, model_name in zip(checkpoints,model_names):
    new_model_path = os.path.join(new_exp_folder, model_name + ".pth")
    torch.save(checkpoint, new_model_path)

# TODO: finish 
def create_experiment_analysis():
  pass