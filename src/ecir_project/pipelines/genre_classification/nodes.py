"""
This is a boilerplate pipeline 'genre_classification'
generated using Kedro 0.17.7
"""
import torch
from torch import nn
import time
from thesis_project.mmnet.data import OneModalEmbeddingDataset, OneModalSampler
from thesis_project.mmnet.evaluation import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from typing import Dict
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import shutil



def create_model(input_size, opt):
    """
    Returns a simple pytorch two layer perceptron with hidden layer size 512 and output layer size 29 (number of XITE genres)
    arguments: input_size: number of input nodes. Depends on the size of the embeddings that are used as input
    """
    class Net(nn.Module):
        def __init__(self, input_size, opt):
            super().__init__()
            # layers
            self.fc1 = nn.Linear(input_size, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 30)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(0)

            # loss and optimizer
            self.criterion = torch.nn.CrossEntropyLoss()
            params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters())
            self.optimizer = torch.optim.Adam(params, lr=opt['learning_rate'])

            # iterations
            self.Eiters = 0


        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.softmax(self.fc3(x))
            return x


    return Net(input_size, opt)

    
def train_model(network: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                opt: Dict,
                modality: str):
    """
    Function that trains a network for genre classification
    arguments:  network: pytorch network object to be trained
                train_loader: kedro lazy dataloader for the training data
                val_loader: kedro lazy dataloader for the validation data
                opt: dictionary of parameters for optimization
    """
    # keep track of the best loss, last epochs loss and of how many times  the loss has increased in a row  

    if torch.cuda.is_available() and opt['use_gpu']:
        network = network.cuda()

    loss_v = 0
    best_loss_v = 1e9
    loss_counter = 0

    # iterate trough epochs
    for epoch in range(opt['max_epochs']):
        print(f'Epoch [{epoch}]')
        network.optimizer.step()

        # train for one epoch
        train_epoch(opt, train_loader, network, epoch)

        # evaluate on validation set
        loss_v = validate(opt, val_loader, network)

        # remember best loss and save checkpoint
        is_best = loss_v < best_loss_v
        best_loss_v = min(loss_v, best_loss_v)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': network.state_dict(),
            'best_rsum': best_loss_v,
            'opt': opt,
            'Eiters': network.Eiters,
        }, is_best, prefix=opt['logger_name'])
        
        
        # increase the counter if loss increased, and possible stop
        if loss_v > best_loss_v:
            loss_counter += 1
            if loss_counter >= opt['max_increases']:
                break
        else:
            loss_counter = 0
            best_loss_v = loss_v

    return network

def train_epoch(opt: Dict,
                train_loader: torch.utils.data.DataLoader,
                network: nn.Module,
                epoch: int):
    run = wandb.init(project="Genre Network ", entity="karelveldkamp")
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    network.train()

    end = time.time()
    for batch in tqdm(train_loader):
        # measure data loading time

        embeddings, idx, genre = batch
        if torch.cuda.is_available() and opt['use_gpu']:
            embeddings = embeddings.cuda()
            genre = genre.cuda()

        data_time.update(time.time() - end)

        network.Eiters += 1
        out = network.forward(embeddings)

        # measure accuracy and record loss
        network.optimizer.zero_grad()
        loss = network.criterion(out, genre)
        loss_meter.update(loss)

        # compute gradient and do SGD step
        loss.backward()
        network.optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # send log to weights and biases
        wandb.log({
            'BatchTime': batch_time.val,
            'DataTime': data_time.val
        })
    # log average loss at the end of epoch
    wandb.log({
        'Train loss': loss_meter.avg,
        'Epoch': epoch
    })

def validate(opt: Dict,
            val_loader:torch.utils.data.DataLoader, 
            network: nn.Module):
    # create trackers for metrics
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    network.eval()

    # loop though batches to compute validation metrics
    for batch in val_loader:
        with torch.no_grad():
            if torch.cuda.is_available() and opt['use_gpu']:
                batch[0] = batch[0].cuda()
                batch[2] = batch[2].cuda() 

            output = network(batch[0])
            # calculate loss and accuracy
            output_labels = torch.argmax(output, axis=1)

            loss = network.criterion(output, batch[2])
            is_equal = output_labels==batch[2]
            acc = torch.mean(is_equal.float())
            
            loss_meter.update(loss)
            acc_meter.update(acc)
            

    wandb.log({
        'Validation loss': loss_meter.avg,
        'Accuracy': acc_meter.avg,
    })

    return loss_meter.avg

def evaluate(network, 
            test_loader,
            opt):
    acc_meter = AverageMeter()
    network.eval()
    for batch in test_loader:
        with torch.no_grad():
            if torch.cuda.is_available() and opt['use_gpu']:
                batch[0] = batch[0].cuda()
                batch[2] = batch[2].cuda()
            output = network(batch[0])
            output_labels = torch.argmax(output, axis=1)
            is_equal = output_labels==batch[2]
            acc_meter.update(torch.mean(is_equal.float()))
            
    
    wandb.log({
        'Test Accuracy': acc_meter.avg,
    })

    return acc_meter.avg



def create_dataloaders(train_loaders: Dict, 
                    val_loaders: Dict,
                    test_loaders: Dict,
                    opt: Dict,
                    metadata):
    """
    function that cerates pytorch dataloaders for the train, validation and test splits
    """
    # create train dataloader
    train_dataset = OneModalEmbeddingDataset(train_loaders,
                                            metadata,
                                            opt['train_size'])
    sampler = OneModalSampler(train_dataset)
    batch_sampler = BatchSampler(sampler, opt['batch_size'], True)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)

    # create validation dataloader
    val_dataset = OneModalEmbeddingDataset(val_loaders,
                                            metadata,
                                            opt['val_size'])
    sampler = OneModalSampler(val_dataset)
    batch_sampler = BatchSampler(sampler, opt['val_batch_size'], True)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler)

    # create train dataloader
    test_dataset = OneModalEmbeddingDataset(test_loaders,
                                            metadata, 
                                            opt['test_size'])
    sampler = OneModalSampler(test_dataset)
    batch_sampler = BatchSampler(sampler, opt['test_batch_size'], True)
    test_dataloader = DataLoader(test_dataset, batch_sampler=batch_sampler)

    return train_dataloader, val_dataloader, test_dataloader

    


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'genre_class_best.pth.tar')



