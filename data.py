# coding: utf-8
# Jiayang Cheng

import torch
from torchvision import datasets, transforms

def get_MNIST_datasets(data_path="./data/", transform=None):
    """ Retrieve the MNIST data. (which DO NOT have a validation set)
    This function will automatically download MNIST data under the ``data_path'' folder.
    The default tranform for the images are used if transform is specified as None.
    """
    if transform is None:
        # Normalize with the global mean and standard deviation of the MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307],std=[0.3081])])

    data_train = datasets.MNIST(root=data_path, transform=transform, train=True, download=True)
    data_test = datasets.MNIST(root=data_path, transform=transform, train=False, download=True)
    return {'train': data_train, 'test': data_test}

def get_MNIST_dataloaders(MNIST_data, train_valid_split=(50000, 10000), train_batch_size=32, eval_batch_size=32, seed=2022, num_workers=4):
    """ Get MNIST dataloaders, including splitting the original train set into train and validation sets.
    """
    train_data, valid_data = torch.utils.data.random_split(MNIST_data['train'], train_valid_split, 
                                                            generator=torch.Generator().manual_seed(seed))
    test_data = MNIST_data['test']

    data_loaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=num_workers),
        'valid': torch.utils.data.DataLoader(valid_data, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
    }

    return data_loaders

def get_GLUE_datasets(dataset_name='sst2'):
    """ Retrieve the given NLP dataset from the GLUE collection.
    See this webpage for dataset descriptions:
    https://huggingface.co/datasets/glue
    """
    from datasets import load_dataset
    glue_dataset = load_dataset("glue", dataset_name)
    return dict(glue_dataset)

def get_GLUE_dataloaders(glue_dataset, train_batch_size=8, eval_batch_size=32, num_workers=4):
    """ Get GLUE dataloaders for a given GLUE dataset (the returned dict from ``get_GLUE_datasets'').
    """
    data_loaders = {}
    for split_name in glue_dataset:
        if split_name.lower() == 'train':
            data_loaders[split_name] = torch.utils.data.DataLoader(glue_dataset[split_name], batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
        else:
            data_loaders[split_name] = torch.utils.data.DataLoader(glue_dataset[split_name], batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    return data_loaders




if __name__ == '__main__':
    """ Get MNIST dataset (image data) """
    MNIST_data = get_MNIST_datasets()
    print(MNIST_data)

    # Get dataloaders, note that here you need to specify the train/valid split ratio, batch_size, 
    # and also don't forget to change the seed
    MNIST_loaders = get_MNIST_dataloaders(MNIST_data, [50000, 10000], 32, 2022, 2)
    for batch in MNIST_loaders['train']:
        print(batch[0].size(), batch[1].size())
        break

    """ Get GLUE dataset (text data) """
    dataset = get_GLUE_datasets('sst2')
    glue_loaders = get_GLUE_dataloaders(dataset, train_batch_size=8, eval_batch_size=32, num_workers=4)
    for batch in glue_loaders['train']:
        print(batch)
        break