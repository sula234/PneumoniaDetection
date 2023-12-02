import logging

import torch
from torch.cuda.amp import GradScaler

import hydra
from omegaconf import DictConfig

from dataset import trainset, testset
from model import CNN
from utils.operations import train, get_model_from_cfg, save_experiment, create_new_experiment
from utils.log import formatter


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:

    # ===================================
    # config
    cfg = cfg.config
    models = cfg.models
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    model_path = cfg.model_folder
    pretrained = cfg.pretrained

    # TODO: add more losses
    loss_fn = torch.nn.CrossEntropyLoss()
    if cfg.loss_fn == 'cross-entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    
    # ===================================

    log = logging.getLogger(__name__)
    log.propagate = False  # Disable propagation for your logger

    # Create a console handler and set the formatter
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)

    #TODO: fix this
    if pretrained:
        checkpoint = torch.load(model_path)

    # # mixed precision setup
    # dtype = "bfloat16"
    # ptdtype = {'float32': torch.float32,
    #            'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    
    checkpoints = []
    for model in models:
        # model
        net = get_model_from_cfg(model)
        
        if pretrained:
            net.load_state_dict(checkpoint["model"])
        
        if net == None:
            log.error("Wrong names for models list in config")
            break

        log.info(f"{model} parameters amount: {net.get_parameters_amount():,}")
        
        optim = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay)

        if pretrained:
            optim.load_state_dict(checkpoint["optimizer"])

        # define dataloaders
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

        checkpoint = train(net=net, train_loader=trainloader, test_loader=testloader, 
            epochs=epochs, loss_fn=loss_fn, optimizer=optim, log=log, model_name=model)
        checkpoints.append(checkpoint)
        log.info(f"{model}'s checkpoint is saved")

    if len(checkpoints) != 0:
        # Write checkpoints as desired, e.g.,
        new_exp_folder = create_new_experiment()
        save_experiment(new_exp_folder=new_exp_folder, model_names=models, checkpoints=checkpoints)
        log.info("Experiment is saved")
    else:
        log.warning("No models in checkpoints")
    


if __name__ == "__main__":
    app()
