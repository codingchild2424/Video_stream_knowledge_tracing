#auto_encoder trainer

import torch
from copy import deepcopy

import numpy as np

from torch.nn.functional import one_hot
from sklearn import metrics

from tqdm import tqdm

class AE_trainer():

    def __init__(self, model, optimizer, device, crit, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.crit = crit
        self.n_epochs = config.n_epochs

        print(self.model)
    
    def _train(self, train_loader):
        for data in tqdm(train_loader):
            self.model.train()
            
            data = data.to(self.device)

            y_hat, verb_emb = self.model( data.long() )

            self.optimizer.zero_grad()

            loss = self.crit( y_hat, verb_emb )

            loss.backward()
            self.optimizer.step()

        return loss

    def _test(self, test_loader):
        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                
                data = data.to(self.device)

                y_hat, verb_emb = self.model( data.long() )

                loss = self.crit( y_hat, verb_emb )

        return loss

    def train(self, train_loader, test_loader):

        best_loss = np.inf
        best_model = None

        for epoch_index in range(self.n_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            train_loss = self._train(train_loader)
            test_loss = self._test(test_loader)

            if test_loss <= best_loss:
                best_loss = test_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d) result: train_loss=%.4f  test_loss=%.4f  best_loss=%.4f" % (
                epoch_index + 1,
                self.n_epochs,
                train_loss,
                test_loss,
                best_loss,
            ))

        print("\n")
        print("The best_loss in Training Session is %.4f" % (
                best_loss,
            ))
        print("\n")
        
        # 가장 최고의 모델 복구    
        self.model.load_state_dict(best_model)