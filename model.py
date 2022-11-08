"""
Contains the implementation of models
"""
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
            self.hidden_dim = self.model.config.d_model  # bart
        except:
            self.hidden_dim = self.model.config.hidden_size  # roberta/bert

        self.classifier = LMClassificationHead(self.hidden_dim,
                                               self.args.num_sup_labels,
                                               classifier_dropout=self.args.classifier_dropout)
        self.self_exp = KFactor(args.num_unsup_clusters, self.hidden_dim,
                                args.num_factor_per_cluster)

    def forward(self):
        raise NotImplementedError('Should use specific functions to feed-forward the model. '
                                  'These functions are ...')

    def get_lm_embedding(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.model_type == "bart":
            # embedding of [EOS] in the decoder
            eos_mask = input_ids.eq(self.model.config.eos_token_id)
            if torch.any(eos_mask.sum(1) > 1):
                raise ValueError("All examples must have only one <eos> tokens.")

            sentence_representation = outputs[0][eos_mask, :].view(outputs[0].size(0), -1,
                                                                   outputs[0].size(-1))[:, -1, :]
        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]

        return sentence_representation

    def get_unsup_loss(self, embs):
        # self expression loss + clustering
        se_loss, cluster_label = 0, []
        for s in ['anchor', 'pos', 'neg']:
            loss, label = self.self_exp(embs[s])
            se_loss += loss
            if s == 'anchor':
                cluster_label = label

        # triplet loss
        triplet_loss = nn.functional.triplet_margin_loss(embs['anchor'], embs['pos'], embs['neg'])

        return se_loss + triplet_loss, cluster_label


class KFactor(nn.Module):
    """
    SelfExpression module, implement K-FACTORIZATION SUBSPACE CLUSTERING introduced in
    https://dl.acm.org/doi/pdf/10.1145/3447548.3467267
    Follow algorithm 4 and 5 in the paper
    """

    def __init__(self, num_clusters, factor_dim=128, num_factor_per_cluster=64):
        super().__init__()
        assert factor_dim > num_factor_per_cluster

        self.num_clusters = num_clusters
        self.factor_dim = factor_dim  # dim of basis vector d^j_i
        self.num_factor_per_cluster = num_factor_per_cluster  # size of basis U^j
        self.D = nn.Parameter(torch.randn((num_clusters, factor_dim, num_factor_per_cluster)))

    def forward(self, x):
        """
        Compute C given D, this process is not involed in backward propagation
        """
        with torch.no_grad():
            Dtx = torch.einsum('nij,bi->nbj', self.D, x)
            DTD_invs = self.compute_DTD_inv()  # compute DTD_invs
            Cs = torch.einsum('nbj,nkj->nbk', Dtx,
                              DTD_invs)  # is that line 9, Algorithm 5, in the paper?
            x_hat = torch.einsum('nij,nbj->nbi', self.D, Cs)
            label = torch.norm(x_hat - x.view(1, -1, self.factor_dim), dim=-1).argmin(dim=0)
            C = Cs[label, [i for i in range(x.shape[0])]]

        se_loss = 0  # TODO: don't we add some constraint on D? e.g L2 regularization loss
        for i in range(x.shape[0]):
            se_loss += torch.sum((x[i] - self.D[label[i]] @ C[i]).square())  # @ is matmul

        return (se_loss, label)

    def compute_DTD_inv(self):
        DTD_inv_list = []  # still take the gradient
        for d in self.D:
            dtd = d.T @ d  # TODO: don't we add gamma*I here?
            DTD_inv_list.append(torch.inverse(dtd))

        return torch.stack(DTD_inv_list, dim=0)


# ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py#L1431
class LMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels=2, classifier_dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
