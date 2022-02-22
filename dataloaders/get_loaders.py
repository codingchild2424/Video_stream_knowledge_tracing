from torch.utils.data import DataLoader, random_split
from utils import collate_fn
from dataloaders.video_stream_loader import Video_stream_loader

from utils import video_stream_collate_fn

#여기서 
def get_video_stream_loaders(config):

    video_stream_dataset = Video_stream_loader()

    longest_sq_len = video_stream_dataset.longest_sq_len

    train_size = int( len(video_stream_dataset) *  config.train_ratio)
    test_size = len(video_stream_dataset) - train_size

    train_dataset, test_dataset = random_split(
        video_stream_dataset, [ train_size, test_size ]
    )

    #train, test 데이터 섞기
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = video_stream_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = video_stream_collate_fn
    )

    return train_loader, test_loader, longest_sq_len

#autoencoder의 차원축소를 위해 데이터를 나누지 않은 것
def get_video_stream_loaders_no_split(config):

    video_stream_dataset = Video_stream_loader()

    longest_sq_len = video_stream_dataset.longest_sq_len

    #train, test 데이터 섞기
    data_loader = DataLoader(
        video_stream_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = video_stream_collate_fn
    )

    return data_loader, longest_sq_len