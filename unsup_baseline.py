from sklearn.cluster import SpectralClustering
import numpy as np
import torch
from tqdm import tqdm

from post_clustering import acc


def unsup_baseline(model, tokenized_data, args, device='cpu'):
    num_samples = len(tokenized_data['test']['label'])
    print(num_samples)
    num_steps_per_epoch = (num_samples - 1) // args.batch_size + 1
    test_embs_for_clustering = []
    index = np.arange(num_samples)
    for i in tqdm(range(num_steps_per_epoch),
                    position=0,
                    total=num_steps_per_epoch):
        # beginning and ending index for this batch
        b, e = i * args.batch_size, min((i + 1) * args.batch_size, num_samples)

        idx = index[b:e]
        input_ids = tokenized_data['test']['text']['input_ids'][idx].to(device)
        attention_mask = tokenized_data['test']['text']['attention_mask'][idx].to(device)


        embs = model.get_lm_embedding(input_ids, attention_mask)
        test_embs_for_clustering.append(embs.cpu().detach())

    test_embs_for_clustering = torch.cat(test_embs_for_clustering, 0).numpy()
    test_clustering = SpectralClustering(n_clusters=args.num_unsup_clusters,
                                    assign_labels='cluster_qr',
                                    random_state=args.seed)
    test_clustering.fit(test_embs_for_clustering)
    test_accuracy = acc(tokenized_data['test']['label'].numpy(), test_clustering.labels_)


    # num_samples = len(tokenized_data['train']['label'])
    # print(num_samples)
    # num_steps_per_epoch = (num_samples - 1) // args.batch_size + 1
    # train_embs_for_clustering = []
    # index = np.arange(num_samples)
    # for i in tqdm(range(num_steps_per_epoch),
    #                 position=0,
    #                 total=num_steps_per_epoch):
    #     # beginning and ending index for this batch
    #     b, e = i * args.batch_size, min((i + 1) * args.batch_size, num_samples)

    #     idx = index[b:e]
    #     input_ids = tokenized_data['train']['text']['input_ids'][idx].to(device)
    #     attention_mask = tokenized_data['train']['text']['attention_mask'][idx].to(device)


    #     embs = model.get_lm_embedding(input_ids, attention_mask)
    #     train_embs_for_clustering.append(embs.cpu().detach())

    # train_embs_for_clustering = torch.cat(train_embs_for_clustering, 0).numpy()
    # train_clustering = SpectralClustering(n_clusters=args.num_unsup_clusters,
    #                                 assign_labels='cluster_qr',
    #                                 random_state=args.seed)
    # train_clustering.fit(train_embs_for_clustering)
    # train_accuracy = acc(tokenized_data['train']['label'].numpy(), train_clustering.labels_)
    # return test_accuracy, train_accuracy, train_embs_for_clustering, train_clustering.labels_
    return test_accuracy
