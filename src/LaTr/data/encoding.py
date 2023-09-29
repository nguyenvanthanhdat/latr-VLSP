from preprocess import preprocess
from tqdm.auto import tqdm

data_dict = preprocess().create_dict_df()
print(data_dict.keys())

class encoding():
    def label2id(self, data_dict):
        label2id = {}   ## Would be responsible for mapping answer to token
        current_word_id = 1
        for split in data_dict.keys():
            raw_answer = data_dict[split]['question'][
                'answers'].values.tolist()

            for ans in tqdm(raw_answer):
                for word in ans.split(" "):

                    if word not in label2id:
                        label2id[word] = {'id':current_word_id, 'count': 1}
                        current_word_id+=1
                    
                    else:
                        label2id[word]['count']+=1
        
        id2label = ["" for _ in range(current_word_id)]
        for key, value in list(label2id.items()):
            id2label[value['id']] = key
        
        return id2label

a = encoding().label2id(data_dict=data_dict)