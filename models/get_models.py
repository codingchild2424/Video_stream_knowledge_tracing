from models.dkt import DKT
from models.ae import AE

def get_AE(config, num_q, device):
    model = AE(
        config,
        num_q = num_q #가장 긴 길이의 sequence
    ).to(device)

    return model

def get_dkt_models(num_q, device, config):

    if config.model_name == "dkt":
        model = DKT(
            num_q = num_q,
            emb_size = config.dkt_emb_size,
            hidden_size = config.dkt_hidden_size
        ).to(device)
    #-> 추가적인 모델 정의
    else:
        print("Wrong model_name was used...")

    return model