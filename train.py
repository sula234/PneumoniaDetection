import logging

import torch
from torch.cuda.amp import GradScaler

import hydra
from omegaconf import DictConfig

from dataset import trainset, testset
from model import Net
from utils.operations import train, validate


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:

    # ===================================
    # config
    cfg = cfg.config
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    model_file = cfg.model_file
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    pretrained = cfg.pretrained
    grad_scaler = cfg.grad_scaler
    loss_fn = torch.nn.CrossEntropyLoss()
    if cfg.loss_fn == 'cross-entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    
    # ===================================

    log = logging.getLogger(__name__)
    if pretrained:
        checkpoint = torch.load(model_file)

    # mixed precision setup
    dtype = "bfloat16"
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # model
    model = Net()
    model = model.to(device)
    #model = torch.compile(model)
    if pretrained:
        model.load_state_dict(checkpoint["model"])
    log.info(f"Model parameters amount: {model.get_parameters_amount():,}")
    
    optim = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=grad_scaler)

    if pretrained:
        optim.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    # define dataloaders
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device={device}')
    train(net=model, train_loader=trainloader, test_loader=testloader, 
          epochs=epochs, loss_fn=loss_fn)
    


if __name__ == "__main__":
    app()
