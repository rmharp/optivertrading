import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn import preprocessing
import os
from tqdm import tqdm



class OptiverDataset(Dataset):
    
    def __init__(self, split=None):
        data = pd.read_csv(f'{os.getcwd()}/optiver-trading-at-the-close/train.csv')
        data = data.dropna()
        mean_scaler = preprocessing.StandardScaler()
        
        relevant_features = ['imbalance_size', 'imbalance_buy_sell_flag','reference_price',
                            'matched_size','far_price','near_price',
                            'bid_price','bid_size','ask_price',
                            'ask_size','wap','time_id',
                            'target']
        
        stock_ids = set(data[["stock_id"]].values.flatten())
        if split == 'train':
            stock_ids = list(stock_ids)[:int(len(stock_ids) * 0.8)]
        elif split == 'val':
            stock_ids = list(stock_ids)[int(len(stock_ids) * 0.8):int(len(stock_ids) * 0.9)]
        elif split == 'test':
            stock_ids = list(stock_ids)[int(len(stock_ids) * 0.9):]
        elif split == 'valid':
            stock_ids = list(stock_ids)[int(len(stock_ids) * 0.8):]

        groups = data.groupby('stock_id')
        
        self.stocks = []
        self.targets = []
        
        progress = tqdm(total=len(stock_ids), desc='Loading data')
        for stock_id in stock_ids:
            stock_data = groups.get_group(stock_id)[relevant_features].sort_values(by=['time_id'])
            target_data = stock_data[["target"]].values
            stock_data = stock_data.drop(columns=['target']).values
            
            stock_data = mean_scaler.fit_transform(stock_data)
            
            self.stocks.append(torch.from_numpy(stock_data))
            self.targets.append(torch.from_numpy(target_data).squeeze())
            
            # if stock_id == 0:
            #     print(self.stocks[0].shape)
            #     print(self.targets[0].shape)
            
            
            
            
            # stock_data = groups.get_group(stock_id)[relevant_features].sort_values(by=['time_id'])
            # target_data = stock_data[["target"]].values.tolist()[lookback - 1:]
            # stock_data = stock_data.drop(columns=['target']).values.tolist()
            
            # stacked_stock_data = []
            # stacked_stock_data += [stock_data[i:i+lookback] for i in range(len(stock_data) - lookback + 1)]
            
            # self.stocks += stacked_stock_data
            # self.targets += target_data
            
            
            
            progress.update()
        progress.close()
        
    def get_tensor(self, l):
        return torch.tensor(l, dtype=torch.float)
    
    def __getitem__(self, index):
        return self.stocks[index].float(), self.targets[index].float()
    
    def __len__(self):
        return len(self.stocks)