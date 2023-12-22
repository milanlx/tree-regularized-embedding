import os 
import sys 
import time 

# t2v 
sys.path.append('/home/ec2-user/SageMaker/VCAC_Multimodel/t2v-main/')

from train_tree import resample
from tabular_mlp import SimpleMLP, VanillaMLP
from tabular_transformer import Config, TabTransformer
from train_nn import ClassifierDataset, create_loss_fn, init_weights
from tree_embedding import XGBSimpleTrees, XGBTrees, TreeToVectorSimple, TreeToVector

# torch 
import torch 
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# torch metrics 
from torchmetrics.classification import AUROC 

# fabric 
import lightning_fabric as lf
from lightning_fabric.loggers import CSVLogger

# args
import argparse

# eval & baseline 
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

# supress warning 
import warnings
warnings.filterwarnings("ignore")


def run(hparams):
    
    ## read data & xgb model 
    xgb_model_path = './model/tree/xgb_model_50_cbcc.pt'
    
    """
    df_train = pd.read_csv('./data/df_sample.csv')
    df_valid = pd.read_csv('./data/df_sample.csv')
    df_test  = pd.read_csv('./data/df_sample.csv' )
    """
    
    df_train = pd.read_csv('./data/df_train.csv')
    df_valid = pd.read_csv('./data/df_valid.csv')
    df_test  = pd.read_csv('./data/df_test.csv' )
    

    ## preprocess data
    # resample training set 
    df_train = resample(df_train, 20)
    
    # prepare for NN model 
    x_train, y_train = df_train.iloc[:,1:].to_numpy(), df_train['tag'].to_numpy()
    x_valid, y_valid = df_valid.iloc[:,1:].to_numpy(), df_valid['tag'].to_numpy()
    x_test,  y_test  = df_test.iloc[:,1:].to_numpy() , df_test['tag'].to_numpy()
    print("read data completed")

    # initialize tree embedder 
    if hparams.tokenizer == 'vector':
        xgbTree = XGBSimpleTrees(xgb_model_path, x_train.shape[1])
    elif hparams.tokenizer == 'sequence':
        xgbTree = XGBTrees(xgb_model_path, x_train.shape[1])
    emb_dim = xgbTree.num_encode
    
    ## train prepare 
    
    # data loader  
    train_data = ClassifierDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    valid_data = ClassifierDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).long())
    test_data  = ClassifierDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(dataset=train_data, batch_size=hparams.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=hparams.batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_data,  batch_size=hparams.batch_size, shuffle=False)
    
    # initialize lightning fabric 
    logger = CSVLogger(root_dir=".", flush_logs_every_n_steps=1)
    fabric = lf.Fabric(accelerator="cuda", devices=4, strategy="ddp", loggers=logger)
    fabric.launch()
    fabric.seed_everything(1337)
    
    # highlight 
    fabric.print(hparams.tokenizer)
    
    # model 
    model_config = Config(n_class=2, n_node=emb_dim, n_layer=2, n_head=4, n_embd=32, n_tree=50, use_pos=True, use_metric=False)
    
    # loss function 
    criterion = create_loss_fn(hparams)
    
    if hparams.tokenizer == 'vector':
        model = SimpleMLP(input_size=emb_dim, output_size=2, emb_dim=256)
        model.apply(init_weights)
        transform_batch = transforms.Compose([TreeToVectorSimple(xgbTree)])
    elif hparams.tokenizer == 'sequence':
        model = TabTransformer(model_config)   
        transform_batch = transforms.Compose([TreeToVector(xgbTree)])
    
    # optimizier 
    #optimizer = optim.Adam(model.parameters(), lr=hparams.lr)
    optimizer = optim.AdamW(model.parameters(), lr=hparams.lr)
    
    # define LR scheduler 
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=60)

    
    # fabric related
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, valid_loader, test_loader = fabric.setup_dataloaders(train_loader, valid_loader, test_loader)
    
    # use torchmetrics for AUC
    valid_auroc = AUROC(task="binary").to(fabric.device)
    test_auroc  = AUROC(task="binary").to(fabric.device)

    best_valid_auc = 0
    
    ## epoch loop 
    for epoch in range(1, hparams.epochs):
        start_time = time.time()
        
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # transform 
            data = transform_batch(data)
            # forward 
            output = model(data)
            loss = criterion(output, target)
            # backward 
            fabric.backward(loss)
            optimizer.step()
        
        execuation_time = time.time() - start_time
        fabric.print(execuation_time)
        
        scheduler.step()
        
        # validation loop 
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data = transform_batch(data)
                output = model(data)
                valid_loss += criterion(output, target)
                # evaluate metric 
                sfmx = nn.Softmax(dim=1)
                output = sfmx(output)
                output = output[:,1]
                valid_auroc(output, target)
 
        valid_loss = fabric.all_gather(valid_loss).sum() / len(valid_loader.dataset)
        valid_auc_score = valid_auroc.compute()
        fabric.print(f"\n valid set: average loss: {valid_loss:.6f}, auc: ({100 * valid_auc_score:.2f}%)\n")
        
        # write log 
        fabric.log_dict({"epoch": epoch, "valid_auc": valid_auc_score, "valid_loss": valid_loss})
        
        # early stopping and save model 
        if valid_auc_score > best_valid_auc:
            best_valid_auc = valid_auc_score
            # testing loop 
            with torch.no_grad():
                for data, target in test_loader:
                    data = transform_batch(data)
                    output = model(data)
                    # evaluate metric 
                    sfmx = nn.Softmax(dim=1)
                    output = sfmx(output)
                    output = output[:,1]
                    test_auroc(output, target)
            test_auc_score = test_auroc.compute()
            fabric.print(f"\n test set: auc: ({100 * test_auc_score:.2f}%)\n")
            # write log 
            fabric.log_dict({"epoch": epoch, "test_auc": test_auc_score})
            
            # save model 
            if fabric.global_rank == 0:
                state_dict = model.state_dict()
                torch.save(state_dict, os.path.join('model/nn', f"epoch-{epoch:06d}-ckpt-model.pth"))
        
        valid_auroc.reset()
        test_auroc.reset()
      

if __name__ == "__main__":    
    # basic setup 
    torch.set_float32_matmul_precision("high")

    # hyper parameters 
    parser = argparse.ArgumentParser(description="DDP T2V Experiments")
    
    # training related 
    parser.add_argument(
        "--batch-size", type=int, default=256, metavar="B", help="input batch size for training (default: 256)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, metavar="LR", help="learning rate (default: 1e-4)")
    parser.add_argument("--epochs", type=int, default=60, metavar="EP", help="number of training epochs (default: 30)")
    # task related 
    parser.add_argument("--num-classes", type=int, default=2, metavar="NC", help="number of class (default: 2)")
    # model related
    parser.add_argument("--tokenizer", type=str, default="sequence", metavar="T2V", help="tree tokenzier (default: sequence)")

    hparams = parser.parse_args()

    run(hparams)