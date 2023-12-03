import torch
from tqdm import tqdm
from model import CNN, MLP, VGG16, PretrainedVGG16
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

EXPs_PATH = "models"

def validate(net,test_loader,device, loss_fn):
  count=acc=0
  running_loss = 0.0
  for xo,yo in tqdm(test_loader,desc="Validation"):
    x,y = xo.to(device), yo.to(device)
    with torch.no_grad():
      p = net.forward(x)

      # get loss 
      loss = loss_fn(p, y)
      running_loss += loss.item()

      # predict
      _,predicted = torch.max(F.softmax(p,1), 1)
      
      acc+=(predicted==y).sum()
      count+=len(x)

  return acc/count, running_loss/len(test_loader)

def train(net,train_loader,test_loader,epochs,loss_fn, optimizer, log, model_name):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  log.info(f'Using device={device}') 
  net = net.to(device)
  
  # ===================================
  # metrics

#   roc_curve_data = {'true': [], 'probs': []}
#   pr_curve_data = {'true': [], 'probs': []}

  metrics = {"losses": {'train': [], 'validate': []},
             "accuracies": {'train': [], 'validate': []},
             "conf_matrix": None}
  # ===================================


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

    # get validation data
    val_acc, val_loss = validate(net,test_loader,device=device, loss_fn=loss_fn)

    log.info(f"Epoch={ep}, Train accuracy={acc/count}, Validation accuracy={val_acc}, Loss={running_loss/len(train_loader)}")

    # save metrics
    metrics["losses"]["validate"].append(val_loss)
    metrics["losses"]["train"].append(running_loss/len(train_loader))

    metrics["accuracies"]["validate"].append(val_acc)
    metrics["accuracies"]["train"].append(acc/count)

  # save model
  checkpoint = {
    "model": net.state_dict(),
    "optimizer": optimizer.state_dict()
    }
  return checkpoint, metrics


def get_model_from_cfg(model: str):
  
  if model.upper() == "CNN":
    return CNN()
  
  elif model.upper() == "MLP":
    return MLP()
  
  elif model.upper() == "VGG16":
    return VGG16()
  
  elif model.upper() == "PRETRAINEDVGG16":
    return PretrainedVGG16()
  
def create_new_experiment():
    # Check if the folder exists
    if not os.path.exists("models"):
        # If it doesn't exist, create it
        os.makedirs("models")
    
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


def save_experiment(new_exp_folder, model_names, checkpoints, metrics):
  
  for checkpoint, model_name, metric in zip(checkpoints, model_names, metrics):
    new_model_path = os.path.join(new_exp_folder, model_name + ".pth")
    torch.save(checkpoint, new_model_path)
    plot_metrics(metric["losses"] , metric["accuracies"], metric["conf_matrix"], new_exp_folder, model_name)

# TODO: finish 
def create_experiment_analysis():
  pass


def plot_metrics(losses, accuracies, conf_matrix, save_path, model_name):
    # Plot loss
    plt.plot(losses['train'], label='Training Loss')
    plt.plot(losses['validate'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_path}/loss_plot_{model_name}.png')
    plt.clf()

    # Plot accuracy
    plt.plot([loss.cpu() for loss in accuracies['validate']], label='Validation Accuracy')
    plt.plot([loss.cpu() for loss in accuracies['train']], label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_path}/accuracy_plot_{model_name}.png')
    plt.clf()

    # Plot confusion matrix
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.clf()


def get_confusion_matrix(net, test_loader, device="cuda"):
    net.eval()
    
    # Move the model to the specified device
    net.to(device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_preds)

    return confusion_mat