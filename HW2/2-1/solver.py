import os
import math
import time
from torch import nn, optim

from utils import *


class Solver(object):
    def __init__(self, model, pad_id):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

        self.model.apply(init_weights)

        print(self.model)

    def _train_epoch(self, data_iter):
        self.model.train()

        # CLIP = 1
        epoch_loss = 0
        for i, (_, feats, captions) in enumerate(data_iter):
            output = self.model(feats, captions)
            output_dim = output.shape[-1]

            loss = self.criterion(output.reshape(-1, output_dim), captions.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(data_iter)
    
    @torch.no_grad()
    def _evaluate_epoch(self, data_iter):
        self.model.eval()

        epoch_loss = 0
        for i, (_, feats, captions) in enumerate(data_iter):
            output = self.model(feats, captions, teacher_forcing_ratio=0)  # turn off teacher forcing
            output_dim = output.shape[-1]

            loss = self.criterion(output.reshape(-1, output_dim), captions.reshape(-1))

            epoch_loss += loss.item()

        return epoch_loss / len(data_iter)

    def train(self, train_iter, valid_iter, num_epochs):
        best_valid_loss = float('inf')

        for epoch in range(num_epochs):
            start_time = time.time()
    
            train_loss = self._train_epoch(train_iter)
            valid_loss = self._evaluate_epoch(valid_iter)
    
            end_time = time.time()
    
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self._save_model('s2vt_model_bestscore.pt')
    
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        # self._save_model('s2vt_model_final.pt')
        
    def evaluate(self, data_iter):
        loss = self._evaluate_epoch(data_iter)
        print(f'| Test Loss: {loss:.3f} | Test PPL: {math.exp(loss):7.3f} |')

    def _save_model(self, save_name, save_path='./saved_model'):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, save_name))
        # print(f'Save model to {save_path}')
