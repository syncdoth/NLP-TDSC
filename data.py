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

def get_MNIST_dataloaders(MNIST_data, train_valid_split=(50000, 10000), batch_size=32, seed=2022, num_workers=4):
    """ Get MNIST dataloaders, including splitting the original train set into train and validation sets.
    """
    train_data, valid_data = torch.utils.data.random_split(MNIST_data['train'], train_valid_split, 
                                                            generator=torch.Generator().manual_seed(seed))
    test_data = MNIST_data['test']

    data_loaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'valid': torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return data_loaders

if __name__ == '__main__':
    # Get MNIST dataset
    MNIST_data = get_MNIST_datasets()
    print(MNIST_data)

    # Get dataloaders, note that here you need to specify the train/valid split ratio, batch_size, 
    # and also don't forget to change the seed
    MNIST_loaders = get_MNIST_dataloaders(MNIST_data, [50000, 10000], 32, 2022, 2)
    for batch in MNIST_loaders['train']:
        print(batch[0].size(), batch[1].size())
        break