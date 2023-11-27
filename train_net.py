from lstm.lstm import OptiverModel
from lstm.data import OptiverDataset
from lstm.neural import OptiverNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

def collate_fn(data):
    return torch.stack([item[0] for item in data]), torch.stack([item[1] for item in data])


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = OptiverModel()
    model = OptiverNet()
    model = model.to(device)
    
    if 'model_net_best.pt' in os.listdir('./lstm/checkpoints'):
        print('found existing model')
        model.load_state_dict(torch.load('./lstm/checkpoints/model_net_best.pt', map_location=device))
    
    batch_size = 1
    
    if 'BATCH_SIZE' in os.environ:
        batch_size = int(os.environ['BATCH_SIZE'])
    
    train_dataset = OptiverDataset(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_dataset = OptiverDataset(split='valid')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    num_epochs = 50
    
    maes = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        model.train()
        progress = tqdm(total=len(train_dataloader), desc='Training')
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x.squeeze())
            
            loss = loss_fn(y_pred.squeeze(), y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix({'loss': loss.item()})
            progress.update()
        progress.close()
        
        torch.save(model.state_dict(), f"./lstm/checkpoints/model_net_epoch{epoch + 1}.pt")
        
        model.eval()
        loss = 0
        num_items = 0
        progress = tqdm(total=len(val_dataloader), desc='Validating')
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x.squeeze())
            loss += torch.sum(torch.absolute(y_pred.squeeze() - y.squeeze())).item()
            num_items += torch.numel(y_pred)
            progress.update()
        progress.close()
        mae = loss/num_items
        maes.append(mae)
        print(f'MAE = {mae}')
        print(f'Best MAE = {min(maes)}')

if __name__ == '__main__':
    main()