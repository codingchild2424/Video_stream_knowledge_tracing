import torch

from define_argparser import define_argparser

from dataloaders.get_loaders import get_video_stream_loaders, get_video_stream_loaders_no_split
from models.get_models import get_AE
from trainers.get_trainers import get_ae_trainers
from utils import get_AE_optimizer, get_AE_crits

def main(config):

    #0. device 선언
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    #Autoencoder

    #1. 데이터 받아오기
    ae_train_loader, ae_test_loader, longest_sq_len = get_video_stream_loaders(config)
    #2. 모델 받아오기
    ae_model = get_AE(config, longest_sq_len, device)
    #3. optimizer 선택
    ae_optimizer = get_AE_optimizer(ae_model, config)
    #4. criterion
    ae_crit = get_AE_crits(config)
    #5. trainer 선택
    ae_trainer = get_ae_trainers(
        model = ae_model,
        optimizer= ae_optimizer,
        device = device,
        crit = ae_crit,
        config = config
    )

    #6. trainer 훈련
    ae_trainer.train(ae_train_loader, ae_test_loader)

    # #7. model 기록 저장 위치
    ae_model_path = './train_model_records/' + 'ae_' + config.model_fn

    #8. model 기록
    torch.save({
        'model': ae_trainer.model.state_dict(),
        'config': config
    }, ae_model_path)

    #######
    #9. autoencoder로 훈련시킨 모델을 활용해서, encoder만을 가져와서 차원을 변환시킨 데이터를 가져옴
    #분리되지 않은 data를 가져오기
    no_split_dataloader, longest_sq_len = get_video_stream_loaders_no_split(config)

    trained_ae_model = get_AE(config, longest_sq_len, device) #모델 구조 선언

    checkpoint = torch.load( ae_model_path ) #학습된 모델의 경로 가져오기

    trained_ae_model.load_state_dict(
        checkpoint['model'] #저장된 모델 중에서 model만 가져오기
    )

    trained_ae_optimizer = get_AE_optimizer(trained_ae_model, config)

    trained_ae_crit = get_AE_crits(config)

    trained_ae_model.eval() #평가모드 활성화

    ae_trainer = get_ae_trainers(
            model = trained_ae_model,
            optimizer= trained_ae_optimizer,
            device = device,
            crit = trained_ae_crit,
            config = config
        )

    dim_reduct_list = ae_trainer.dim_reductor(no_split_dataloader)

    print("dim_reduct_list: ", dim_reduct_list)

    #여기서 나온 값을 정오답 값으로 만들어진 one-hot벡터와 같이 concat하기

    #정오답 값으로 one_hot 만들기

    #concat

    #dkt 모델 훈련



    # #6. train
    # trainer.train(train_loader, test_loader)





    #1. 데이터 받아오기
    #train_loader, test_loader, num_q = get_loaders(config)
    
    #2. model 선택
    # model = get_models(num_q, device, config)
    
    # #3. optimizer 선택
    # optimizer = get_optimizers(model, config)
    
    # #4. criterion 선택
    # crit = get_crits(config)
    
    # #5. trainer 선택
    # trainer = get_trainers(model, optimizer, device, num_q, crit, config)

    # #6. 훈련 및 score 계산
    # y_true_record, y_score_record = trainer.train(train_loader, test_loader)

    # #7. model 기록 저장 위치
    # model_path = './train_model_records/' + config.model_fn

    # #8. model 기록
    # torch.save({
    #     'model': trainer.model.state_dict(),
    #     'config': config
    # }, model_path)

    # #9. 시각화 결과물 만들기
    # get_visualizers(y_true_record, y_score_record,model, model_path, test_loader, device, config)

#main
if __name__ == "__main__":
    config = define_argparser() #define_argparser를 불러옴
    main(config)