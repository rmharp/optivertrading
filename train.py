from lstm.lstm import OptiverModel
from lstm.data import OptiverDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

def collate_fn(data):
    return torch.stack([item[0] for item in data]), torch.stack([item[1].squeeze() for item in data])


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OptiverModel()
    model = model.to(device)
    
    batch_size = int(os.environ['BATCH_SIZE'])
    lookback = int(os.environ['LOOKBACK'])
    train_dataset = OptiverDataset(lookback=lookback, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_dataset = OptiverDataset(lookback=lookback, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    progress = tqdm(total=len(train_dataloader), desc='Training')
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.update()
    progress.close()
    torch.save(model.state_dict(), './lstm/checkpoints/model.pth')
    
    model.eval()
    loss = 0
    progress = tqdm(total=len(val_dataloader), desc='Validating')
    for x, y in val_dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss += torch.sum(torch.absolute(y_pred - y)).item()
        progress.update()
    progress.close()
    print(loss/len(val_dataset))  

if __name__ == '__main__':
    main()