from src.LaTr.config.config import ConfigurationManager
import json, os
import pandas as pd
import copy
from tqdm.auto import tqdm

config_data = ConfigurationManager().config.data
# print(config_data)
tqdm.pandas()

class preprocess():
    def create_dict_df(self) -> dict:

        dict_data = {
            'train': self.read2df(config_data.train),
            'val': self.read2df(config_data.val)
        }

        # check file name in dataframe exist or not
        ## train dataframe
        dict_data['train']['question']['path_exists'] = dict_data[
            'train']['question']['image_id'].progress_apply(
            lambda x: os.path.exists(os.path.join(config_data.image_path, x)+'.jpg'))

        dict_data['train']['question'] = dict_data['train']['question'][
            dict_data['train']['question']['path_exists']==True]

        ## validation dataframe
        dict_data['val']['question']['path_exists'] = dict_data[
            'val']['question']['image_id'].progress_apply(
            lambda x: os.path.exists(os.path.join(config_data.image_path, x)+'.jpg'))

        dict_data['val']['question'] = dict_data['val']['question'][
            dict_data['val']['question']['path_exists']==True]
        
        dict_data['train']['question'].drop(columns = ['flickr_original_url', 'flickr_300k_url','image_classes', 'question_tokens', 'path_exists'
                              ], axis = 1, inplace = True)
        dict_data['val']['question'].drop(columns = ['flickr_original_url', 'flickr_300k_url','image_classes', 'question_tokens', 'path_exists'
                              ], axis = 1, inplace = True)

        return dict_data

    def read2df(self, split):
        # data_path = config_data.split
        # dict_data = config_data.to_dict()
        dict_data = split.to_dict()
        dfs = {}
        for type_dataset in dict_data:
            temp = pd.DataFrame(json.load(open(dict_data[type_dataset]))['data'])
        # dict_data['question'] = pd.DataFrame(json.load(open(dict_data['question']))['data'])
        # print(dict_data)
            if type_dataset == 'question':
                temp['answers'] = temp['answers'].apply(
                    lambda x: " ".join(list(map(str, x))))
            # print(type(temp))
            dfs[type_dataset] = temp
        return dfs

# base_img_path = os.path.join('drive/MyDrive/TextVQA', 'train_images')
# train_json_df['path_exists'] = train_json_df['image_id'].progress_apply(
#     lambda x: os.path.exists(os.path.join(base_img_path, x)+'.jpg'))
# train_json_df = train_json_df[train_json_df['path_exists']==True]

# val_json_df['path_exists'] = val_json_df['image_id'].progress_apply(
#     lambda x: os.path.exists(os.path.join(base_img_path, x)+'.jpg'))
# val_json_df = val_json_df[val_json_df['path_exists']==True]

