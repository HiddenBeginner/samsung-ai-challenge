from .utils import AverageMeter

import time
import torch


class RGCNSolver:
    def __init__(self, model, lr, n_epochs, device=None):
        self.model = model
        self.device = device
        self.n_epochs = n_epochs
        
        self.criterion = torch.nn.L1Loss()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=lr)
        self.model.to(self.device)
        self.history = {'train_loss': [], 'valid_loss': [], 'dev_loss': []}
    
    def fit(self, train_loader, valid_loader, dev_loader):
        for epoch in range(self.n_epochs):
            # training phase
            t = time.time()
            train_loss = self.train_one_epoch(train_loader)
            
            # validation phase
            valid_loss = self.evaluate(valid_loader)
            dev_loss = self.evaluate(dev_loader)
            
            message = f"[Epoch {epoch}] "
            message += f"Elapsed time: {time.time() - t:.3f} | "
            message += f"Train loss: {train_loss.avg:.5f} | "
            message += f"Validation loss: {valid_loss.avg:.5f} | "
            message += f"Dev loss: {dev_loss.avg:.5f} |"
            print(message)
            self.history['train_loss'].append(train_loss.avg)
            self.history['valid_loss'].append(valid_loss.avg)
            self.history['dev_loss'].append(dev_loss.avg)
             
    def train_one_epoch(self, loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        
        for step, data in enumerate(loader):
            print(
                f'Train step {(step + 1)} / {len(loader)} | ' +
                f'Summary loss: {summary_loss.avg:.5f} | ' +
                f'Time: {time.time() - t:.3f} |', end='\r'
            )
            data.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(data.x, data.edge_index, data.edge_type, data.batch)
            loss = self.criterion(y_pred, data.y.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            
            summary_loss.update(loss.detach().item(), data.num_graphs)
            
        return summary_loss
    
    def evaluate(self, loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        
        with torch.no_grad():
            for step, data in enumerate(loader):
                data.to(self.device)
                
                y_pred = self.model(data.x, data.edge_index, data.edge_type, data.batch)
                loss = self.criterion(y_pred, data.y.unsqueeze(1))
                summary_loss.update(loss.detach().item(), data.num_graphs)
                
        return summary_loss
