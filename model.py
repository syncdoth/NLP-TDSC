"""
Contains the implementation of models
"""

from collections import OrderedDict
import torch
from torch import nn
from transformers import AutoModel


class TdscLanguageModel(nn.Module):
    """
    This is the implementation for TDSC Language Model, which is used to train on NLP tasks,
    based on stem models from Huggingface and algorithms from our proposal
        model: stem Language Model to produce contextualized embedding
        classifier: head Fc layer to map contextualized embedding to logits
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.model_type = self.model.config.model_type
        try:
            self.emb_size = self.model.config.d_model # bart
        except:
            self.emb_size = self.model.config.hidden_size # roberta/bert

        self.classifier = LMClassificationHead(self.emb_size, self.args.num_labels)

    def forward(self):
        raise NotImplementedError(
            'Should use specific functions to feed-forward the model. '
            'These functions are ...'
        )

    def get_lm_embedding(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.model_type == "bart":
            # embedding of [EOS] in the decoder
            eos_mask = input_ids.eq(self.model.config.eos_token_id)
            if torch.any(eos_mask.sum(1) > 1):
                raise ValueError("All examples must have only one <eos> tokens.")

            sentence_representation = outputs[0][eos_mask, :].view(
                outputs[0].size(0), -1, outputs[0].size(-1))[:, -1, :]

        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]
        
        return sentence_representation


# ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py#L1431
class LMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels=2, classifier_dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, embs, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

