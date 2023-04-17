from transformers import EarlyStoppingCallback, IntervalStrategy
import transformers

from typing import List, Optional, Tuple, Union

import os

import torch
import torch.nn as nn

import numpy as np

from siVAEtorch.trainer.VAE_trainer import CustomTrainer

def train_model(
    model: Optional[torch.nn.Module],
    MODEL_DIR: Optional[str],
    trainset,
    validset,
    batch_size: Optional[int] = 16,
    num_train_epochs: Optional[int] = 100,
    return_trainer: Optional[bool] = False,
    early_stopping_patience: Optional[int] = 10,
    dataloader_num_workers: Optional[int] = 0,
    logging_dir: Optional[str] = None,
    **kwargs,
  ):

    ## Load model if exists
    if os.path.isfile(os.path.join(MODEL_DIR,'pytorch_model.bin')):
        state_dict = torch.load(os.path.join(MODEL_DIR,'pytorch_model.bin'))
        model.load_state_dict(state_dict)

    else:

        ## Set up training arguments
        trainingarguments = transformers.training_args.TrainingArguments(output_dir=MODEL_DIR,
                                                                        per_device_train_batch_size=batch_size,
                                                                        per_device_eval_batch_size=batch_size,
                                                                        dataloader_pin_memory=False,
                                                                        num_train_epochs=num_train_epochs,
                                                                        load_best_model_at_end=True,
                                                                        metric_for_best_model = 'loss',
                                                                        evaluation_strategy='epoch',
                                                                        save_strategy='epoch',
                                                                        dataloader_num_workers=dataloader_num_workers,
                                                                        logging_dir=logging_dir,
                                                                        logging_first_step=True,
                                                                        logging_strategy='epoch',
                                                                        **kwargs,
                                                                        )

        # trainingarguments = transformers.training_args.TrainingArguments(output_dir=MODEL_DIR,
        #                                                                 per_device_train_batch_size=batch_size,
        #                                                                 per_device_eval_batch_size=batch_size,
        #                                                                 dataloader_pin_memory=False,
        #                                                                 num_train_epochs=num_train_epochs,
        #                                                                 load_best_model_at_end=False,
        #                                                                 dataloader_num_workers=dataloader_num_workers,
        #                                                                 logging_dir=logging_dir,
        #                                                                 **kwargs,
        #                                                                 )

        ## Create trainer
        es = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        callbacks = [es]

        # callbacks = []

        trainer = CustomTrainer(model=model,
                                train_dataset=trainset,
                                eval_dataset=validset,
                                args = trainingarguments,
                                callbacks = callbacks,
        )

        ## Train model
        try:
            train_output = trainer.train(resume_from_checkpoint=True)
        except:
            train_output = trainer.train(resume_from_checkpoint=False)

        ## Save model
        trainer.save_model()

        ## Remove checkpoint models
        import shutil
        from glob import glob

        checkpoints = glob(os.path.join(MODEL_DIR,"checkpoint-*"))

        # Save trainer_state from final checkpoint
        def get_final_checkpoint(checkpoints):
            def get_step(checkpoint):
                return int(os.path.basename(checkpoint).lstrip('checkpoint-'))
            steps = [get_step(cp) for cp in checkpoints]
            return checkpoints[np.argmax(steps)]

        checkpoint_final = get_final_checkpoint(checkpoints)
        os.rename(os.path.join(checkpoint_final,'trainer_state.json'),
                  os.path.join(MODEL_DIR,'trainer_state.json'))

        # Remove checkpoint models
        _ = [shutil.rmtree(c) for c in checkpoints]

    return model
