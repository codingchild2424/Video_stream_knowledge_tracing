from trainers.ae_trainer import AE_trainer
from trainers.dkt_trainer import DKT_trainer
from trainers.ae_trainer import AE_trainer

def get_ae_trainers(model, optimizer, device, crit, config):
    trainer = AE_trainer(
        model = model,
        optimizer = optimizer,
        device = device,
        crit = crit,
        config = config
    )

    return trainer

def get_trainers(model, optimizer, device, num_q, crit, config):

    #trainer 실행
    if config.model_name == "dkt":
        trainer = DKT_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit
        )

    return trainer
