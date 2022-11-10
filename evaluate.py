"""
implementation of evaluation loop, loss, and metrics functions
"""
import torch
from tqdm import tqdm


def evaluate(model, eval_data, loss_fn, args, device='cpu'):
    """ Evaluates a given model and dataset. 
    Obtained from:
    https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/master/src/evaluate.py
    """
    model.eval()
    sample_count = 0
    running_loss = 0
    running_acc = 0
    num_eval_samples = eval_data[2]
    num_eval_steps = (num_eval_samples-1)//args.batch_size + 1
    # TODO: add other metric, e.g AUC, Macro F1

    with torch.no_grad():

        for i in tqdm(range(num_eval_steps), position=0, total=num_eval_steps, desc=f'Evaluation'):
            b, e = i*args.batch_size, min((i+1)*args.batch_size, num_eval_samples)
            idx = range(b,e)
            input_ids = eval_data['text']['input_ids'][idx].to(device)
            attention_mask = eval_data['text']['attention_mask'][idx].to(device)
            labels = eval_data['label'][idx].to(device)

            embs = model.get_lm_embedding(input_ids, attention_mask)
            logits = model.classifier(embs)
            loss = loss_fn(logits, labels)

            sample_count += input_ids.size(0)
            running_loss += loss.item() * input_ids.size(0)             # smaller batches count less
            running_acc += (logits.argmax(-1) == labels).sum().item()   # num corrects

        loss = running_loss / sample_count
        acc = running_acc / sample_count

    return loss, acc
