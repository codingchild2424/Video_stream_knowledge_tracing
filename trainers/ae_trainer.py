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
            mask_seqs = mask_seqs.unsqueeze(-1).expand(
                mask_seqs.size()[0], 
                mask_seqs.size()[1], 
                self.config.ae_emb_size
                )

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
                mask_seqs = mask_seqs.unsqueeze(-1).expand(
                    mask_seqs.size()[0], 
                    mask_seqs.size()[1], 
                    self.config.ae_emb_size
                    )

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

    #encoder용
    def dim_reductor(self, data_loader):

        dim_reduct_list = []

        with torch.no_grad():
            for data in tqdm(data_loader):
                self.model.eval()

                batches, mask_seqs = data
                batches = batches.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                mask_seqs = mask_seqs.unsqueeze(-1).expand(
                    mask_seqs.size()[0], 
                    mask_seqs.size()[1], 
                    self.config.ae_emb_size
                    )

                y_hat = self.model.dim_reductor( batches.long() )

                #y_hat = torch.masked_select(y_hat, mask_seqs)

                #torch.masked_select를 사용하면 결과 값의 차원이 1차원으로 변환됨
                #원하는 것은 차원이 달라지지 않은 형태의 벡터이므로, 
                
                y_hat = y_hat * mask_seqs # 이렇게 하면 곱은 가능하지만, 0인 부분이 삭제되지는 않음

                #https://stackoverflow.com/questions/61956893/how-to-mask-a-3d-tensor-with-2d-mask-and-keep-the-dimensions-of-original-vector

                #y_hat의 절대값(abs) 상태에서 마지막 차원의 차원(2) 방향으로 모두 더한 값이 0보다 큰지 아닌지를 담은 벡터
                #https://stackoverflow.com/questions/60888546/how-can-i-remove-elements-across-a-dimension-that-are-all-zero-with-pytorch
                nonZeroRows = torch.abs(y_hat).sum(dim=2) > 0

                #이렇게 대입하면, 2차원 형태의 값을 얻을 수 있음
                y_hat = y_hat[nonZeroRows] #ex) torch.Size([142, 50])

                dim_reduct_list.append(y_hat)

        return dim_reduct_list



        #최종적인 return 값은 차원이 축소된 벡터들임
