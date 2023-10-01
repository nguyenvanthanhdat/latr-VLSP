from src.LaTr.data.preprocess import preprocess
from src.LaTr.data.encoding import encoding
from src.LaTr.model import model
from src.LaTr.data.dataset import dataset
from src.LaTr.config.config import ConfigurationManager
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb, torch
import pytorch_lightning as pl


# device = 'cpu'
# finetuned_latr = model.LaTr_for_finetuning(config_model).to(device)

def main():
    config_data = ConfigurationManager().config.data
    data_dict = preprocess(config_data).create_dict_df()
    label2id, id2label, current_word_id = encoding().label2id(data_dict)
    data_module = dataset(
        config_data = config_data,
        data_dict = data_dict,
        label2id = label2id,
        current_word_id = current_word_id
    ).get_dataset()
    config_model = {
        't5_model': 't5-base',
        'vocab_size': 32128,
        'hidden_state': 768,
        'max_2d_position_embeddings': 1001,
        'classes': current_word_id,
        'seq_len': 512
    }
    url_for_ckpt = 'https://www.kaggleusercontent.com/kf/99663112/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..GfuZWkqwWi9nROCTnAS3OQ.YowTb3CNlES2WS_F6BvOSrGs3uLWc2kSBkhElYUcndML0Feuiizdu8trA2e4aj_kdluv1nYlVpS3_86VaJfgSBtyJShQoB0CyxCqdvdMiKl4eQQdWUv2XrTBecEJPXupdFaElzr57CcRjpz35rueyDjf3GVJLznkpSdoyWwSxoxCACbUpS73PKWi97WHfPmEWQgXTDxT_Uno_Pau6fayKyzJ-vWrETzOA2Z6f1-i7umK48D7JBQacS2g_40dW8wIH34QsztCZhHOake7qZnXU_19qaFeDQCNldZ4HcGAmKMtqYI_NK_By370IZ6OHe5Q-mh1f_9SaZoXCzzgaNx4Wsw1THZgzSjZgP2dTLP6a4ZkjHFWiZdkl0azvmoCmSVVYbRdQ9_iI9sFvhUpDWj1bOlr-Zrq9gRi8ksaH9rIzrzk63x_fKPGphZKpxB_l_6iewdGt4yb3GB8kWyGrxBnsGvV5Ei7gTaqv9OAkSKTACMEKB-rj-T8HKtk3ktnEqGMCpHTpkB8RYE6EqYRPbnSYMShjZb12GSn5uYntLtcG7MUbQX-OMt0vzh9fag_zpCyO89K56jxZ6Q9kWdADG0C2T0nR8uC8vWUUBptWNc2tt6pcupcUO19kt7ddNHMbxajHym5AijizrfJbkqnujEodlHWc8C77PawpX2xUPvIlbSvhbdsRRyYfOFGLmZsDdKa.c9dgiKXE5w_-qo4J3He6Qw/models/epoch=0-step=34602.ckpt'
    # datamodule = DataModule(train_ds, val_ds)
    max_steps = 50000       ## 60K Steps
    latr = model.LaTrForVQA(config_model, max_steps= max_steps)
    
    # try:
    #     latr = latr.load_from_checkpoint(url_for_ckpt)
    #     print("Checkpoint loaded correctly")
    # except:
    #     print("Could not load checkpoint")
    #     return 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_ce_loss", mode="min"
    )
    
    wandb.init(config=config_model, project="VQA with LaTr")
    # wandb_logger = WandbLogger(project="VQA with LaTr", log_model = True, entity="iakarshu")
    wandb_logger = WandbLogger(project="VQA with LaTr", log_model = True, entity="iakarshu")
    
    ## https://www.tutorialexample.com/implement-reproducibility-in-pytorch-lightning-pytorch-lightning-tutorial/
    pl.seed_everything(42, workers=True)
    
    trainer = pl.Trainer(
        max_steps = max_steps,
        default_root_dir="logs",
        # gpus=(1 if torch.cuda.is_available() else 0),
#         accelerator="tpu",
#         devices=8,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=True
    )
    
    trainer.fit(latr, data_module)

if __name__ == "__main__":
    main()