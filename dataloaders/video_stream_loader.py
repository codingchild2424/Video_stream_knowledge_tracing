from torch.utils.data import Dataset

import pandas as pd
import numpy as np

#dummy_data
DATASET_DIR = 'datasets/deidentified_datasets.csv'

class Video_stream_loader(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir

        self.final_video_stream_logs, self.video_name_logs, \
            self.answer_logs, self.longest_sq_len = self.preprocessor(self.dataset_dir)
        
    def __getitem__(self, index):
        #출력되는 벡터는 가장 긴 벡터를 기준으로 길이가 맞춰져있고, 만약 빈칸이 있는 데이터의 경우에는 -1로 채워져있음
        return self.final_video_stream_logs[index]

    def __len__(self):
        return len(self.final_video_stream_logs)

    #여기서 가장 긴 길이
    def preprocessor(self, dataset_dir):

        pd_data = pd.read_csv(dataset_dir)

        #uniq 한 값
        uniq_actor = np.unique(pd_data['actor'])
        uniq_obj = np.unique(pd_data['object'])
        uniq_verb = np.unique(pd_data['verb'])

        #딕셔너리에 각 값에 대한 인덱스 짝을 적어두기
        obj2idx = { obj : idx + 1 for idx, obj in enumerate(uniq_obj) } #idx에 +1을 하는 것은 나중에 빈 값에 0을 넣기 위해서임
        verb2idx = { verb: idx + 1 for idx, verb in enumerate(uniq_verb) }

        print('obj2idx: ', obj2idx)
        print('verb2idx: ', verb2idx)

        start_idx = verb2idx['시청기록']
        correct_idx = verb2idx['퀴즈 정답']
        wrong_idx = verb2idx['퀴즈 오답']

        #각 object를 숫자로 모두 차환
        for idx, obj in enumerate(uniq_obj):
            pd_data.replace(obj, idx + 1, inplace = True)

        #각 verb값을 숫자로 모두 치환
        for idx, verb in enumerate(uniq_verb):
            pd_data.replace(verb, idx + 1, inplace = True)

        #숫자로 변환된 리스트 데이터
        uniq_actor_idx = np.unique(pd_data['actor'])
        uniq_obj_idx = np.unique(pd_data['object'])
        uniq_verb_idx = np.unique(pd_data['verb'])

        #학생별로 학습한 비디오의 이름을 담는 리스트
        video_name_logs = [] #[ [name1, name2, ...], [name3, name4 ....] ]
        #학생별로 풀이한 퀴즈의 정오답 값이 담긴 리스트
        answer_logs = []  #[ [0, 1, ....], [1, 0, ...] ]
        #학생별로 풀이한 비디오에 대한 로그
        video_stream_log_total = []

        for idx in uniq_actor_idx: #user 한명씩 가져오기
            pd_data_actor = pd_data[pd_data['actor'] == idx] #user별 데이터프레임

            video_name_log = []
            answer_log = []
            video_stream_logs = []

            for obj in uniq_obj_idx: #각 비디오 하나씩 가져오기
                pd_data_obj = pd_data_actor[pd_data_actor['object'] == obj] #각 유저의 비디오별 데이터 프레임

                #verb에 start_idx이 포함되어있지 않으면 pass
                if (pd_data_obj['verb'] != start_idx).any():
                    pass
                #verb에 correct_idx, wrong_idx 둘 다 포함되어있지 않으면 pass
                if (pd_data_obj['verb'] != correct_idx).any():
                    if(pd_data_obj['verb'] != wrong_idx).any():
                        pass

                video_stream_log = []

                #verb에 0과 5나 6 중 하나가 있다면 실행
                for row in pd_data_obj.itertuples(): #한 행씩 실행
                    #한줄씩 가져오도록 하고, 시청기록(0)부터 퀴즈정오답(5, 6)이 나오기 전까지 수집
                    #print('row', row) #row Pandas(Index=0, _1=0, actor=0, object=0, verb=0, timestamp='2022-02-06T17:18:25.268')

                    #answer
                    if row[4] == correct_idx: #퀴즈 오답은 0을 더하기
                        answer_log.append(0)
                        video_name_log.append(row[3])
                        break
                    elif row[4] == wrong_idx: #퀴즈 정답은 1을 더하기
                        answer_log.append(1)
                        video_name_log.append(row[3])
                        break
                    else:
                        video_stream_log.append(row[4]) #아니면 그냥 video_stream_log에 더하기

                video_stream_logs.append(video_stream_log)


            video_name_logs.append(video_name_log)
            #학생별로 풀이한 퀴즈의 정오답 값이 담긴 리스트
            answer_logs.append(answer_log)
            #학생별로 풀이한 비디오에 대한 로그
            video_stream_log_total.append(video_stream_logs)

            longest_sq_len = 0

            #여기에서 먼저 가장 긴 길이 추출하기
            for video_stream_log in video_stream_log_total:
                for data in video_stream_log:
                    if longest_sq_len < len(data):
                        longest_sq_len = len(data)

            final_video_stream_logs = []

            for video_stream_log in video_stream_log_total:
                for data in video_stream_log:
                    final_video_stream_logs.append(data)

            """
            final_video_stream_logs:  
            [
                [0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 7, 1],
                [0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 4], [0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 7],
                []
            ]
            """

        return final_video_stream_logs, video_name_logs, answer_logs, longest_sq_len