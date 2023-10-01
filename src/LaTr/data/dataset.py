# class decoding():
#     return a
from src.LaTr.config.config import ConfigurationManager
from src.LaTr.data.dataloader import TextVQA
from src.LaTr.data.encoding import encoding
from src.LaTr.data.preprocess import preprocess
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import ViTFeatureExtractor, ViTModel
from transformers import ViTImageProcessor, ViTModel
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader



class dataset():
    # config_data = ConfigurationManager().config.data
    # data_dict = preprocess().create_dict_df()
    # label2id = encoding().label2id()
    # current_word_id = label2id.current_word_id
    def __init__(
            self,
            config_data, 
            data_dict, 
            label2id, 
            current_word_id
    ):
        self.config_data = dict(config_data), 
        self.data_dict = dict(data_dict), 
        self.label2id = label2id, 
        self.current_word_id = current_word_id
        
     

    def get_dataset(self):
        base_img_path = self.config_data[0]['image_path']
        train_json_df = self.data_dict[0]['train']['question']
        val_json_df = self.data_dict[0]['val']['question']
        train_ocr_json_df = self.data_dict[0]['train']['ocr']
        val_ocr_json_df = self.data_dict[0]['val']['ocr']
        t5_model = "t5-base"
        tokenizer = T5Tokenizer.from_pretrained(t5_model)
        max_seq_len = 512
        target_size = (500,384)
        label2id = self.label2id[0]

        # choose your transform method
        # transformer = T5Tokenizer.from_pretrained(t5_model)
        transformer = None


        train_ds = TextVQA(base_img_path = base_img_path,
                        json_df = train_json_df,
                        ocr_json_df = train_ocr_json_df,
                        tokenizer = tokenizer,
                        label2id = label2id,
                        transform = transformer, 
                        max_seq_length = max_seq_len, 
                        target_size = target_size
                        )


        val_ds = TextVQA(base_img_path = base_img_path,
                        json_df = val_json_df,
                        ocr_json_df = val_ocr_json_df,
                        tokenizer = tokenizer,
                        label2id = label2id,
                        transform = transformer, 
                        max_seq_length = max_seq_len, 
                        target_size = target_size
                        )
        return DataModule(train_ds, val_ds)

    def get_current_word_id(self):
        current_word_id = self.label2id.current_word_id
        return current_word_id


class DataModule(pl.LightningDataModule):

        def __init__(self, train_dataset, val_dataset,  batch_size = 2):

            super(DataModule, self).__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.batch_size = batch_size

        def collate_fn(self, data_bunch):
            '''
            A function for the dataloader to return a batch dict of given keys

            data_bunch: List of dictionary
            '''
            vit_feat_extract = ViTImageProcessor("google/vit-base-patch16-224-in21k")
            dict_data_bunch = {}

            for i in data_bunch:
                for (key, value) in i.items():
                    if key not in dict_data_bunch:
                        dict_data_bunch[key] = []
                    dict_data_bunch[key].append(value)

            for key in list(dict_data_bunch.keys()):
                dict_data_bunch[key] = torch.stack(dict_data_bunch[key], axis = 0)

            if 'img' in dict_data_bunch:
                ## Pre-processing for ViT
                dict_data_bunch['img'] = vit_feat_extract(list(dict_data_bunch['img']),return_tensors = 'pt')['pixel_values']

            return dict_data_bunch
        
        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size = self.batch_size, 
                            collate_fn = self.collate_fn, shuffle = True)
        
        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size = self.batch_size,
                                        collate_fn = self.collate_fn, shuffle = False)

# a = dataset()
# b = a.get_dataset()