from transformers import Trainer

from torch.distributions.kl import kl_divergence as kl_div

import torch
from torch import nn

import numpy as np
from tqdm import tqdm

def KL_loss(prior,posterior):
    kl_loss = kl_div(prior,posterior)
    return kl_loss

def Recon_loss(X_dist, X):
    """ NLL of predicted gene expression distribution """
    return -X_dist.log_prob(X)

class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs["labels"]

        # input_ids = torch.squeeze(inputs['input_ids'],dim=1)
        # attention_mask = torch.squeeze(inputs['attention_mask'],dim=1)

        outputs = model(inputs)

        # Sum across output features, average across samples
        recon_loss = Recon_loss(outputs.decoder.dist, inputs['X']).sum(-1).mean()
        # Sum across latent features, average across samples
        kl_loss    = KL_loss(model.encoder.prior, outputs.encoder.dist).sum(-1).mean()
        # loss = recon_loss + kl_loss
        loss = recon_loss + kl_loss

        return (loss, outputs) if return_outputs else loss


    def compute_metrics(self, model, inputs, return_outputs=False):

        labels = inputs["labels"]

        outputs = model(inputs)

        # Sum across output features, average across samples
        recon_loss = Recon_loss(outputs.decoder.dist, inputs['X']).sum(-1).mean()
        # Sum across latent features, average across samples
        kl_loss    = KL_loss(model.encoder.prior, outputs.encoder.dist).sum(-1).mean()
        # loss = recon_loss + kl_loss
        loss = recon_loss + kl_loss

        return {'Total': loss,
                'KL loss': kl_loss,
                'Recon loss': recon_loss}


def evaluate(dataset, model, batch_size=16, num_workers=1, **kwargs):

    # Create dataloaders
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             **kwargs)

    dataiter = iter(dataloader)

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
