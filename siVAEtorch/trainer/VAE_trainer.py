import torch
from torch import nn

from transformers import Trainer
from transformers.utils import ModelOutput

import numpy as np
from tqdm import tqdm

class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs["labels"]

        # input_ids = torch.squeeze(inputs['input_ids'],dim=1)
        # attention_mask = torch.squeeze(inputs['attention_mask'],dim=1)

        outputs = model(inputs)

        try:
            loss_fct = model.loss_fct
        except:
            loss_fct = nn.CrossEntropyLoss()

        logits = outputs.get("logits")

        if model.output_size > 1:
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


    def compute_metrics(self, model, inputs, return_outputs=False):

        labels = inputs["labels"]

        # input_ids = torch.squeeze(inputs['input_ids'],dim=1)
        # attention_mask = torch.squeeze(inputs['attention_mask'],dim=1)

        outputs = model(inputs)

        try:
            loss_fct = model.loss_fct
        except:
            loss_fct = nn.CrossEntropyLoss()

        logits = outputs.get("logits")

        if model.output_size > 1:
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            raise Exception(f"{logits.get_device()},{labels.get_device()}")
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return {'nloss': -loss}


def evaluate(dataset, model, batch_size=16, num_workers=1, **kwargs):

    # Create dataloaders
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             **kwargs)

    labels_all = []
    outputs_all = []
    id_all = []

    with torch.no_grad():

        for inputs in tqdm(iter(dataloader)):

            labels = inputs["label"]
            id = inputs['label_id']

            # # forward pass
            outputs = model(inputs)
            logits = outputs.get("logits")

            # forward pass
            # outputs_bert = model.bert(input_ids=input_ids,
            #                 attention_mask=attention_mask,
            #                 )
            # logits      = model.classifier(outputs_bert['pooler_output'])

            labels_all.append(labels.detach().cpu().numpy())
            outputs_all.append(logits.detach().cpu().numpy())
            id_all.append(id.detach().cpu().numpy())

        labels_all = np.concatenate(labels_all)
        outputs_all = np.concatenate(outputs_all)
        id_all = np.concatenate(id_all)

    return labels_all, outputs_all, id_all
