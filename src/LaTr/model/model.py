import torch.nn as nn
from src.LaTr.data.encoding import encoding
from src.LaTr.utils.utils import LaTr_for_pretraining
import torch
from transformers import ViTModel
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

class LaTr_for_finetuning(nn.Module):
  def __init__(self, config, address_to_pre_trained_weights = None):
    super(LaTr_for_finetuning, self).__init__()

    self.config = config
    self.vocab_size = config['vocab_size']
    self.question_emb = nn.Embedding(config['vocab_size'], config['hidden_state'])

    self.pre_training_model = LaTr_for_pretraining(config)

    if address_to_pre_trained_weights is not None:
      self.pre_training_model.load_state_dict(torch.load(address_to_pre_trained_weights))
    self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
  
    ## In the fine-tuning stage of vit, except the last layer, all the layers were freezed

    self.classification_head = nn.Linear(config['hidden_state'], config['classes'])

  def forward(self, lang_vect, spatial_vect, quest_vect, img_vect):

    ## The below block of code calculates the language and spatial featuer
    embeded_feature =     self.pre_training_model.language_emb(lang_vect)
    top_left_x_feat =     self.pre_training_model.top_left_x(spatial_vect[:,:, 0])
    top_left_y_feat =     self.pre_training_model.top_left_y(spatial_vect[:,:, 1])
    bottom_right_x_feat = self.pre_training_model.bottom_right_x(spatial_vect[:,:, 2])
    bottom_right_y_feat = self.pre_training_model.bottom_right_y(spatial_vect[:,:, 3])
    width_feat =          self.pre_training_model.width_emb(spatial_vect[:,:, 4])
    height_feat =         self.pre_training_model.height_emb(spatial_vect[:,:, 5])

    spatial_lang_feat = embeded_feature + top_left_x_feat + top_left_y_feat + bottom_right_x_feat + bottom_right_y_feat + width_feat + height_feat

    ## Extracting the image feature, using the Vision Transformer
    img_feat = self.vit(img_vect).last_hidden_state
    
    ## Extracting the question vector
    quest_feat = self.question_emb(quest_vect)

    ## Concating the three features, and then passing it through the T5 Transformer
    final_feat = torch.cat([img_feat, spatial_lang_feat,quest_feat ], axis = -2)

    ## Passing through the T5 Transformer
    for layer in self.pre_training_model.list_encoder:
        final_feat = layer(final_feat)[0]

    final_feat = self.pre_training_model.residue_encoder(final_feat)

    for layer in self.pre_training_model.list_decoder:
        final_feat = layer(final_feat)[0]
    final_feat = self.pre_training_model.residue_decoder(final_feat)

    answer_vector = self.classification_head(final_feat)[:, :self.config['seq_len'], :]

    return answer_vector


class LaTrForVQA(pl.LightningModule):
  def __init__(self, config , learning_rate = 1e-4, max_steps = 100000//2):
    super(LaTrForVQA, self).__init__()
    
    self.config = config
    self.save_hyperparameters()
    self.latr =  LaTr_for_finetuning(config)
    self.training_losses = []
    self.validation_losses = []
    self.validation_step_outputs = []
    self.max_steps = max_steps

#   def configure_optimizers(self):
#     optimizer = torch.optim.AdamW(self.parameters(), lr = self.hparams['learning_rate'])
#     warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period = 1000)  
#     scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,total_iters  = self.max_steps,  verbose = True)
#     return [optimizer], [{"scheduler": (lr_scheduler, warmup_scheduler), "interval": "step"}]

#   def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
#         lr_scheduler, warmup_scheduler = scheduler
#         with warmup_scheduler.dampening():
#                 lr_scheduler.step()

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr = self.hparams['learning_rate'])

  def polynomial(base_lr, iter, max_iter = 1e5, power = 1):
    return base_lr * ((1 - float(iter) / max_iter) ** power)

  def forward(self, batch_dict):
    boxes =   batch_dict['boxes']
    img =     batch_dict['img']
    question = batch_dict['question']
    words =   batch_dict['tokenized_words']
    answer_vector = self.latr(lang_vect = words, 
                               spatial_vect = boxes, 
                               img_vect = img, 
                               quest_vect = question
                               )
    return answer_vector


  def calculate_acc_score(self, pred, gt):
      
      ## Function ignores the calculation of padding part
      ## Shape (seq_len, seq_len)
      mask = torch.clamp(gt, min = 0, max = 1)
      last_non_zero_argument = (mask != 0).nonzero()[1][-1]
      pred = pred[:last_non_zero_argument]
      gt = gt[:last_non_zero_argument]  ## Include all the arguments till the first padding index
      
      return accuracy_score(pred, gt)
  
  def calculate_metrics(self, prediction, labels):

      ## Calculate the accuracy score between the prediction and ground label for a batch, with considering the pad sequence
      batch_size = len(prediction)
      ac_score = 0

      for (pred, gt) in zip(prediction, labels):
        ac_score+= self.calculate_acc_score(pred.detach().cpu(), gt.detach().cpu())
      ac_score = ac_score/batch_size
      return ac_score

  def training_step(self, batch, batch_idx):
    answer_vector = self.forward(batch)

    ## https://discuss.huggingface.co/t/bertformaskedlm-s-loss-and-scores-how-the-loss-is-computed/607/2
    loss = nn.CrossEntropyLoss()(answer_vector.reshape(-1,self.config['classes']), batch['answer'].reshape(-1))
    _, preds = torch.max(answer_vector, dim = -1)

    ## Calculating the accuracy score
    train_acc = self.calculate_metrics(preds, batch['answer'])
    train_acc = torch.tensor(train_acc)

    ## Logging
    self.log('train_ce_loss', loss,prog_bar = True)
    self.log('train_acc', train_acc, prog_bar = True)
    self.training_losses.append(loss.item())

    return loss

  def validation_step(self, batch, batch_idx):
    logits = self.forward(batch)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1,self.config['classes']), batch['answer'].reshape(-1))
    _, preds = torch.max(logits, dim = -1)
    ## Validation Accuracy
    val_acc = self.calculate_metrics(preds.cpu(), batch['answer'].cpu())
    val_acc = torch.tensor(val_acc)

    ## Logging
    self.log('val_ce_loss', loss, prog_bar = True)
    self.log('val_acc', val_acc, prog_bar = True)
    
    self.validation_step_outputs.append({'val_loss': loss, 'val_acc': val_acc})
    return {'val_loss': loss, 'val_acc': val_acc}
  ## For the fine-tuning stage, Warm-up period is set to 1,000 steps and again is linearly decayed to zero, pg. 12, of the paper
  ## Refer here: https://github.com/Lightning-AI/lightning/issues/328#issuecomment-550114178
  
  def polynomial(self, base_lr, iter, max_iter = 1e5, power = 1):
    return base_lr * ((1 - float(iter) / max_iter) ** power)
  
  def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure = None, on_tpu=False,
    using_native_amp=False, using_lbfgs=False):

        ## Warmup for 1000 steps
        if self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        ## Linear Decay
        else:
            for pg in optimizer.param_groups:
                pg['lr'] = self.polynomial(self.hparams.learning_rate, self.trainer.global_step, max_iter = self.max_steps)

        optimizer.step(opt_closure)
        optimizer.zero_grad()

  # def on_validation_epoch_end(self, outputs):
        
        
  #       val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
  #       val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

  #       self.log('val_loss_epoch_end', val_loss, on_epoch=True, sync_dist=True)
  #       self.log('val_acc_epoch_end', val_acc, on_epoch=True, sync_dist=True)
        
  #       self.val_prediction = []
  def on_validation_epoch_end(self):
        
        outputs = self.validation_step_outputs
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log('val_loss_epoch_end', val_loss, on_epoch=True, sync_dist=True)
        self.log('val_acc_epoch_end', val_acc, on_epoch=True, sync_dist=True)
        
        self.val_prediction = []
        self.validation_step_outputs.clear()

