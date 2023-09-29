from src.LaTr.config.config import ConfigurationManager
import json, os
import pandas as pd
import copy

config_data = ConfigurationManager().config.data
# print(config_data)


def read2df(split):
    # data_path = config_data.split
    # dict_data = config_data.to_dict()
    dict_data = split.to_dict()
    for type_dataset in dict_data:
        temp = dict_data[type_dataset]
        temp = pd.DataFrame(json.load(open(temp))['data'])
    # dict_data['question'] = pd.DataFrame(json.load(open(dict_data['question']))['data'])
    # print(dict_data)
        if type_dataset == 'question':
            temp['answers'] = temp['answers'].apply(
                lambda x: " ".join(list(map(str, x))))
        # print(type(temp))
    return temp

dict_data = {
    'train': read2df(config_data.train),
    'val': read2df(config_data.val)
}

print(type(dict_data['train']['question']))
# dict_data['train']['question']['path_exists'] = dict_data['train']['question']['image_id'].progress_apply(
#     lambda x: os.path.exists(os.path.join(config_data.image_path, x)+'.jpg'))

# base_img_path = os.path.join('drive/MyDrive/TextVQA', 'train_images')
# train_json_df['path_exists'] = train_json_df['image_id'].progress_apply(
#     lambda x: os.path.exists(os.path.join(base_img_path, x)+'.jpg'))
# train_json_df = train_json_df[train_json_df['path_exists']==True]

# val_json_df['path_exists'] = val_json_df['image_id'].progress_apply(
#     lambda x: os.path.exists(os.path.join(base_img_path, x)+'.jpg'))
# val_json_df = val_json_df[val_json_df['path_exists']==True]

