import torch
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using device={device}')

def validate(net,test_loader):
  count=acc=0
  for xo,yo in tqdm(test_loader):
    x,y = xo.to(device), yo.to(device)
    with torch.no_grad():
      p = net.forward(x)
      _,predicted = torch.max(p,1)
      acc+=(predicted==y).sum()
      count+=len(x)
  return acc/count

def train(net,train_loader,test_loader,epochs,loss_fn):
  net = net.to(device)
  optimizer = torch.optim.Adam(net.parameters())
  for ep in range(epochs):
    count=acc=0
    for xo,yo in tqdm(train_loader):
      x = xo.to(device)
      y = yo.to(device)
      optimizer.zero_grad()
      p = net.forward(x)
      loss = loss_fn(p,y)
      loss.backward()
      optimizer.step()
      _,predicted = torch.max(p,1)
      acc+=(predicted==y).sum()
      count+=len(x)
    val_acc = validate(net,test_loader)
    print(f"Epoch={ep}, train_acc={acc/count}, val_acc={val_acc}")
