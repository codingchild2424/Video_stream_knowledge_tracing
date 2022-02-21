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
        self.config = config

        print(self.model)
    
    def _train(self, train_loader): #train_mask를 활용해서 mask를 씌워줘야 함
        for data in tqdm(train_loader):
            self.model.train()

            batches, mask_seqs = data
            
            batches = batches.to(self.device)
            mask_seqs = mask_seqs.to(self.device)

            #mask와 y_hat, verb_emb의 사이즈가 다르기 때문에, unsqueeze해서 가장 아래 차원을 늘리고, expand를 사용하여 같은 차원으로 만들어 줌
            mask_seqs = mask_seqs.unsqueeze(-1).expand(mask_seqs.size()[0], mask_seqs.size()[1], self.config.ae_emb_size)

            y_hat, verb_emb = self.model( batches.long() )

            self.optimizer.zero_grad()

            #마스크를 씌워서 실제로 가지고 있는 값에 대해서만 계산할 것임
            y_hat = torch.masked_select(y_hat, mask_seqs)
            verb_emb = torch.masked_select(verb_emb, mask_seqs)

            loss = self.crit( y_hat, verb_emb )

            loss.backward()
            self.optimizer.step()

        return loss

    def _test(self, test_loader):
        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                
                batches, mask_seqs = data
            
                batches = batches.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                 #mask와 y_hat, verb_emb의 사이즈가 다르기 때문에, unsqueeze해서 가장 아래 차원을 늘리고, expand를 사용하여 같은 차원으로 만들어 줌
                mask_seqs = mask_seqs.unsqueeze(-1).expand(mask_seqs.size()[0], mask_seqs.size()[1], self.config.ae_emb_size)

                y_hat, verb_emb = self.model( batches.long() )

                y_hat = torch.masked_select(y_hat, mask_seqs)
                verb_emb = torch.masked_select(verb_emb, mask_seqs)

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