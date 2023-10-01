import numpy as np
import torch
import math
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageDraw
import pytesseract
import torch.nn as nn
from transformers import T5ForConditionalGeneration

PAD_TOKEN_BOX = [0, 0, 0, 0] # add to config

def resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h):
    x_scale = target_w / orig_w
    y_scale = target_h / orig_h
    orig_left, orig_top, orig_right, orig_bottom = bbox
    x = int(np.round(orig_left * x_scale))
    y = int(np.round(orig_top * y_scale))
    xmax = int(np.round(orig_right * x_scale))
    ymax = int(np.round(orig_bottom * y_scale))
    return [x, y, xmax, ymax]

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    
    modified from answer here: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    # angle = np.deg2rad(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

def convert_ques_to_token(question, tokenizer, pad_token_id = 0, max_seq_len = 512):

  question_array = []
  question = question.split(" ")
  
  for token in question:
    question_array.extend(tokenizer(token, add_special_tokens = False).input_ids)
  
  if len(question_array)< max_seq_len:
        question_array.extend([pad_token_id]* (max_seq_len-len(question_array)))

  question_array = torch.tensor(question_array, dtype = torch.int32)
  return question_array[:max_seq_len]

def convert_ans_to_token(answer, label2id, max_seq_length = 512 ):

  ## Simple Trick to pad a sequence to deired length
  dummy_array = torch.zeros(max_seq_length)
  actual_ans_array = []
  answer = answer.split(" ")
  for token in answer:
    actual_ans_array.append(label2id[token]['id'])
  
  actual_ans_array = torch.tensor(actual_ans_array, dtype = torch.int32)
  actual_ans_array = pad_sequence([actual_ans_array,dummy_array], batch_first  = True)[0]

  return actual_ans_array

def get_topleft_bottomright_coordinates(df_row):
    left, top, width, height = df_row["left"], df_row["top"], df_row["width"], df_row["height"]
    return [left, top, left + width, top + height]

def apply_ocr(tif_path):
    """
    Returns words and its bounding boxes from an image
    """
    img = Image.open(tif_path).convert("RGB")

    ocr_df = pytesseract.image_to_data(img, output_type="data.frame")
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    words = list(ocr_df.text.apply(lambda x: str(x).strip()))
    actual_bboxes = ocr_df.apply(get_topleft_bottomright_coordinates, axis=1).values.tolist()

    # add as extra columns
    assert len(words) == len(actual_bboxes)
    return {"words": words, "bbox": actual_bboxes}

def get_tokens_with_boxes(unnormalized_word_boxes, list_of_words, tokenizer, pad_token_id = 0, pad_token_box = [0, 0, 0, 0], max_seq_len = 512):
    
    '''
    This function returns two items:
    1. unnormalized_token_boxes -> a list of len = max_seq_len, containing the boxes corresponding to the tokenized words, 
                                    one box might repeat as per the tokenization procedure
    2. tokenized_words -> tokenized words corresponding to the tokenizer and the list_of_words
    '''

    assert len(unnormalized_word_boxes) == len(list_of_words), "Bounding box length!= total words length"
    
    length_of_box = len(unnormalized_word_boxes)
    unnormalized_token_boxes = []
    tokenized_words = []
    
    idx = 0 
    for box, word in zip(unnormalized_word_boxes, list_of_words):
      
      if idx != 0:
        word = " " + word
        
      current_tokens = tokenizer(word, add_special_tokens = False).input_ids
      unnormalized_token_boxes.extend([box]*len(current_tokens))
      tokenized_words.extend(current_tokens)
      idx += 1

    if len(unnormalized_token_boxes)<max_seq_len:
        unnormalized_token_boxes.extend([pad_token_box] * (max_seq_len-len(unnormalized_token_boxes)))
        
    if len(tokenized_words)< max_seq_len:
        tokenized_words.extend([pad_token_id]* (max_seq_len-len(tokenized_words)))
        
    return unnormalized_token_boxes[:max_seq_len], tokenized_words[:max_seq_len]

def create_features(
    img_path,
    tokenizer,
    target_size = (1000, 1000),
    max_seq_length=512,
    use_ocr = True,
    bounding_box = None,
    words = None,
    ):
  
  '''
  We assume that the bounding box provided are given as per the image scale (i.e not normalized), so that we just need to scale it as per the ratio
  '''

  img = Image.open(img_path).convert("RGB")
  width_old, height_old = img.size
  img = img.resize(target_size)
  width, height = img.size
  
  ## Rescaling the bounding box as per the image size
  

  if (use_ocr == False) and (bounding_box == None or words == None):
    raise Exception('Please provide the bounding box and words or pass the argument "use_ocr" = True')

  if use_ocr == True:
    entries = apply_ocr(img_path)
    bounding_box = entries["bbox"]
    words = entries["words"]
  
  bounding_box = list(map(lambda x: resize_align_bbox(x,width_old,height_old, width, height), bounding_box))
  boxes, tokenized_words = get_tokens_with_boxes(unnormalized_word_boxes = bounding_box,
                                               list_of_words = words, 
                                               tokenizer = tokenizer,
                                               pad_token_id = 0,
                                               pad_token_box = PAD_TOKEN_BOX,
                                               max_seq_len = max_seq_length
                                               )


  return img, boxes, tokenized_words

class LaTr_for_pretraining(nn.Module):
    def __init__(self, config, classify = False):

      super(LaTr_for_pretraining, self).__init__()
      self.vocab_size = config['vocab_size']

      model = T5ForConditionalGeneration.from_pretrained(config['t5_model'])
      dummy_encoder = list(nn.Sequential(*list(model.encoder.children())[1:]).children())   ## Removing the Embedding layer
      dummy_decoder = list(nn.Sequential(*list(model.decoder.children())[1:]).children())   ## Removing the Embedding Layer

      ## Using the T5 Encoder

      self.list_encoder = nn.Sequential(*list(dummy_encoder[0]))
      self.residue_encoder = nn.Sequential(*list(dummy_encoder[1:]))
      self.list_decoder = nn.Sequential(*list(dummy_decoder[0]))
      self.residue_decoder = nn.Sequential(*list(dummy_decoder[1:]))

      ## We use the embeddings of T5 for encoding the tokenized words
      self.language_emb = nn.Embedding.from_pretrained(model.shared.weight)  

      self.top_left_x = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.bottom_right_x = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.top_left_y = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.bottom_right_y = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.width_emb = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])
      self.height_emb = nn.Embedding(config['max_2d_position_embeddings'], config['hidden_state'])

      self.classify = classify
      self.classification_layer = nn.Linear(config['hidden_state'], config['classes'])