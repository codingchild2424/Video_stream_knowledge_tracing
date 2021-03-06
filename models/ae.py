from torch.nn import Module, LSTM, Embedding

#autoencoder 형태로 구성
#단 encoder, decoder는 rnn 모델로 구성

class AE(Module):
    
    def __init__(
        self,
        config,
        num_q,
        n_layers=4,
        dropout_p=.2,
    ):
        super().__init__()

        self.emb_size = config.ae_emb_size
        self.hidden_size = config.ae_hidden_size #50, bottle_neck size
        self.num_q = num_q
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        #각 동사의 갯수에 맞게 Embedding layer 갯수를 구성
        self.emb_layer = Embedding(
            self.num_q, self.emb_size
        )
        
        self.encoder = LSTM(
            input_size = self.emb_size, #emb_size
            hidden_size = self.hidden_size, #50, bottle_neck size
            batch_first = True,
            num_layers = self.n_layers,
            dropout = self.dropout_p
        )
        self.decoder = LSTM(
            input_size = self.hidden_size, #50, bottle_neck size
            hidden_size = self.emb_size, #emb_size
            batch_first = True,
            num_layers = self.n_layers,
            dropout = self.dropout_p
        )
        
    def forward(self, x):
        #|x| = (bs, sq, length)
        '''
        train_data:  
            tensor([
                [-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.],
                [0., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 4., 1., 7.]
            ])
        '''
        verb_emb = self.emb_layer(x)
        #print('verb_emb: ', verb_emb)

        """
        verb_emb:  tensor([[[ 0.5335, -0.3515, -0.2204,  ..., -1.2217,  1.1474,  0.0440],
         [ 0.5335, -0.3515, -0.2204,  ..., -1.2217,  1.1474,  0.0440],
         [ 0.5335, -0.3515, -0.2204,  ..., -1.2217,  1.1474,  0.0440],
         ...,
         [ 0.5335, -0.3515, -0.2204,  ..., -1.2217,  1.1474,  0.0440],
         [ 0.5335, -0.3515, -0.2204,  ..., -1.2217,  1.1474,  0.0440],
         [ 0.5335, -0.3515, -0.2204,  ..., -1.2217,  1.1474,  0.0440]],

        [[ 0.5335, -0.3515, -0.2204,  ..., -1.2217,  1.1474,  0.0440],
         [ 0.4171, -0.4438,  0.6080,  ..., -0.2068, -0.6565,  1.5679],
         [ 0.4331,  1.5183, -0.6088,  ...,  0.0635,  0.8479, -0.0905],
         ...,
         [ 0.5949,  0.2352, -0.3215,  ...,  0.7706,  0.2199,  0.1664],
         [ 0.4171, -0.4438,  0.6080,  ..., -0.2068, -0.6565,  1.5679],
         [ 1.0721,  0.7776, -0.1273,  ..., -0.1284, -1.2821, -1.0917]]],
       device='cuda:0', grad_fn=<EmbeddingBackward0>)
        """

        z, _ = self.encoder( verb_emb ) #rnn의 결과값이 두개가 나오므로, 하나를 제거

        #print('z: ', z)

        """
        z:  tensor([[[ 0.0445, -0.0322, -0.0400,  ...,  0.0141,  0.0112,  0.0239],
         [ 0.0683, -0.0392, -0.0660,  ...,  0.0073,  0.0210,  0.0463],
         [ 0.0749, -0.0438, -0.0816,  ...,  0.0018,  0.0270,  0.0584],
         ...,
         [ 0.0933, -0.0511, -0.0902,  ...,  0.0045,  0.0220,  0.0981],
         [ 0.0866, -0.0508, -0.0912,  ..., -0.0034,  0.0220,  0.0960],
         [ 0.1016, -0.0560, -0.0854,  ..., -0.0055,  0.0255,  0.0849]],

        [[ 0.0432, -0.0343, -0.0432,  ...,  0.0144,  0.0147,  0.0271],
         [ 0.0736, -0.0459, -0.0655,  ...,  0.0147,  0.0212,  0.0439],
         [ 0.0819, -0.0470, -0.0815,  ...,  0.0020,  0.0201,  0.0628],
         ...,
         [ 0.0833, -0.0375, -0.0873,  ..., -0.0073,  0.0227,  0.0912],
         [ 0.0899, -0.0445, -0.0872,  ..., -0.0144,  0.0324,  0.0859],
         [ 0.0905, -0.0429, -0.0942,  ..., -0.0109,  0.0326,  0.0935]]],
       device='cuda:0', grad_fn=<CudnnRnnBackward0>)
        """

        y, _ = self.decoder(z)

        #print('y: ', y)

        """
        y:  tensor([[[-0.0257, -0.0273, -0.0247,  ...,  0.0244, -0.0081, -0.0540],
         [-0.0329, -0.0418, -0.0307,  ...,  0.0317, -0.0128, -0.0749],
         [-0.0469, -0.0522, -0.0217,  ...,  0.0381, -0.0045, -0.0880],
         ...,
         [-0.0447, -0.0602, -0.0127,  ...,  0.0356, -0.0085, -0.1045],
         [-0.0440, -0.0560, -0.0057,  ...,  0.0400, -0.0067, -0.1083],
         [-0.0402, -0.0561, -0.0064,  ...,  0.0384, -0.0054, -0.1050]],

        [[-0.0243, -0.0247, -0.0220,  ...,  0.0297, -0.0085, -0.0490],
         [-0.0379, -0.0408, -0.0380,  ...,  0.0371, -0.0069, -0.0707],
         [-0.0388, -0.0551, -0.0344,  ...,  0.0386, -0.0236, -0.0875],
         ...,
         [-0.0455, -0.0640, -0.0071,  ...,  0.0392, -0.0214, -0.1097],
         [-0.0485, -0.0630, -0.0026,  ...,  0.0366, -0.0241, -0.1072],
         [-0.0482, -0.0476,  0.0023,  ...,  0.0401, -0.0186, -0.0950]]],
        device='cuda:0', grad_fn=<CudnnRnnBackward0>)
        """
        
        return y, verb_emb

    #나중에 사용시에는 encoder만 활용하면 됨
    #https://dodonam.tistory.com/301
    def dim_reductor(self, x):

        verb_emb = self.emb_layer(x)

        z, _ = self.encoder( verb_emb )

        return z