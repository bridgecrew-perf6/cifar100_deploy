import argparse
import json
import logging
import os
import sys
import pickle

#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
import torch.utils.data as data


import os
import math
import random as rn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    model = CIFARClassifier(batch_size=batch_size)
    
    if torch.cuda.device_count() > 1:
        print(f"Multi GPU -- {torch.cuda.device_count()}")
        model = nn.DataParallel(model)
        
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        print(f"Loaded Model successfully -- {model_dir}" )
        model.load_state_dict(torch.load(f))
        
    return(model.to(device))

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFARClassifier(pl.LightningModule):
    def __init__(self, train_data_dir=None, batch_size=128, num_workers=4):
        '''Constructor method 
        Parameters:
        train_data_dir (string): path of training dataset to be used either for training and validation
        batch_size (int): number of images per batch. Defaults to 128.
        test_data_dir (string): path of testing dataset to be used after training. Optional.
        num_workers (int): number of processes used by data loader. Defaults to 4.
        '''


        # Invoke constructor
        super(CIFARClassifier, self).__init__()
        
        # Set up class attributes
        self.batch_size = batch_size
        self.train_data_dir = train_data_dir
        self.test_data_dir = train_data_dir
        self.num_workers = num_workers
        
        self.model_ft = models.resnet34(pretrained=True)
        self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 100)
        
    def load_split_train_test(self, valid_size=.2):
        '''Loads data and builds training/validation dataset with provided split size
        Parameters:
        valid_size (float): the percentage of data reserved to validation
        Returns:
        (torch.utils.data.DataLoader): Training data loader
        (torch.utils.data.DataLoader): Validation data loader
        (torch.utils.data.DataLoader): Test data loader
        '''

        num_workers = self.num_workers
        
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        

        # Create transforms for data augmentation. Since we don't care wheter numbers are upside-down, we add a horizontal flip,
        # then normalized data to PyTorch defaults
        train_transforms = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        
        test_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        
        train_data = torchvision.datasets.ImageFolder(root=self.train_data_dir+'/train', transform=train_transforms)
        test_data = torchvision.datasets.ImageFolder(root=self.test_data_dir+'/test', transform=test_transforms)
        

        # loads image indexes within dataset, then computes split and shuffles images to add randomness

        # extracts indexes for train and validation, then builds a random sampler
        #train_idx, val_idx = indices[split:], indices[:split]

        

        train_data_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,  num_workers=num_workers)
        
        
        

        # if testing dataset is defined, we build its data loader as well
        test_loader = None
        if self.test_data_dir is not None:
            test_data_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=num_workers) 
        return train_data_loader, test_data_loader, test_data_loader

    def prepare_data(self):
        '''Prepares datasets. Called once per training execution
        '''
        self.train_loader, self.val_loader, self.test_loader = self.load_split_train_test()
        
    def train_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Training set data loader
        '''
        return self.train_loader

    def val_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Validation set data loader
        '''
        return self.val_loader

    def test_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Testing set data loader
        '''
        return self.test_loader
    
    
    def forward(self, x):
        '''Forward pass, it is equal to PyTorch forward method. Here network computational graph is built
        Parameters:
        x (Tensor): A Tensor containing the input batch of the network
        Returns: 
        An one dimensional Tensor with probability array for each input image
        '''
        
        x = self.model_ft(x)
        return F.log_softmax(x, dim=1)
    
    
    def configure_optimizers(self):
        '''
        Returns:
        (Optimizer): Adam optimizer tuned wit model parameters
        '''
        #return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=1e-3,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return {"optimizer": optimizer}
    
    
    def training_step(self, batch, batch_idx):
        '''Called for every training step, uses NLL Loss to compute training loss, then logs and sends back 
        logs parameter to Trainer to perform backpropagation
        '''

        # Get input and output from batch
        x, labels = batch
        #x, labels = x.to(self.device), labels.to(self.device)
        

        # Compute prediction through the network
        prediction = self.forward(x)

        loss = F.nll_loss(prediction, labels.type(torch.LongTensor).cuda())

        # Logs training loss
        logs = {'train_loss': loss}

        output = {
            # This is required in training to be used by backpropagation
            'loss': loss,
            # This is optional for logging pourposes
            'log': logs
        }

        return output
    
    
    def test_step(self, batch, batch_idx):
        '''Called for every testing step, uses NLL Loss to compute testing loss
        '''
        # Get input and output from batch
        x, labels = batch
        #x, labels = x.to(self.device), labels.to(self.device)

        # Compute prediction through the network
        prediction = self.forward(x)

        loss = F.nll_loss(prediction, labels.type(torch.LongTensor).cuda())

        # Logs training loss
        logs = {'train_loss': loss}

        output = {
            # This is required in training to be used by backpropagation
            'loss': loss,
            # This is optional for logging pourposes
            'log': logs
        }

        return output

    
    def validation_step(self, batch, batch_idx):
        ''' Prforms model validation computing cross entropy for predictions and labels
        '''
        x, labels = batch
        

        
        
        prediction = self.forward(x)
        preds = torch.argmax(prediction, dim=1)
        
        return {
            'val_loss': F.nll_loss(prediction, labels.type(torch.LongTensor).cuda()),
            'val_acc': accuracy(preds, labels.type(torch.LongTensor).cuda())
        }
    

    def validation_epoch_end(self, outputs):
        '''Called after every epoch, stacks validation loss
        '''
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def validation_end(self, outputs):
        '''Called after validation completes. Stacks all testing loss and computes average.
        '''
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Average training loss: '+str(avg_loss.item()))
        logs = {'val_loss': avg_loss}
        return {
            'avg_val_loss': avg_loss,
            'log': logs
        }

    def testing_step(self, batch, batch_idx):
        ''' Prforms model testing, computing cross entropy for predictions and labels
        '''
        x, labels = batch
        prediction = self.forward(x)
        preds = torch.argmax(prediction, dim=1)
        
        return {
            'test_loss': F.nll_loss(prediction, labels.type(torch.LongTensor)),
            'test_acc': accuracy(preds, labels.type(torch.LongTensor).cuda())
        }

    def testing_epoch_end(self, outputs):
        '''Called after every epoch, stacks testing loss
        '''
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def testing_end(self, outputs):
        '''Called after testing completes. Stacks all testing loss and computes average.
        '''
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        print('Average testing loss: '+str(avg_loss.item()))
        logs = {'test_loss': avg_loss}
        return {
            'avg_test_loss': avg_loss,
            'log': logs
        }


if __name__ == "__main__":
    
    

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpus', type=int, default=1) # used to support multi-GPU or CPU training

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('-m','--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('-tr','--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('-te','--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    print(args)

    # Now we have all parameters and hyperparameters available and we need to match them with sagemaker 
    # structure. default_root_dir is set to out_put_data_dir to retrieve from training instances all the 
    # checkpoint and intermediary data produced by lightning
    CIFARTrainer=pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, 
                            accelerator="gpu", strategy="dp",
                            default_root_dir=args.output_data_dir)
    
    
    model = CIFARClassifier(
        batch_size=args.batch_size, 
        train_data_dir=args.train)
    
    # Runs model training 
    CIFARTrainer.fit(model)
    
    # After model has been trained, save its state into model_dir which is then copied back to S3
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)