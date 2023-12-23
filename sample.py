import os 
import sys 
import torch 
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader

sys.path.append('src/')
from train_tree import *
from tree_embedding import TreeToVector, TreeToVectorConverter, TreeToToken, TreeToTokenConverter


class ClassifierDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_batch = self.X[idx]
        Y_batch = self.Y[idx]
        return X_batch, Y_batch


if __name__ == "__main__":
    ## generate synthetic data 
    n_sample, n_dim = 500, 50
    X_train, Y_train = np.random.normal(size=(n_sample, n_dim)), np.random.randint(2, size=n_sample)
    X_valid, Y_valid = np.random.normal(size=(n_sample, n_dim)), np.random.randint(2, size=n_sample)
    

    ## train xgboost model 
    train_data = convert_to_DMatrix(X_train, Y_train)
    valid_data = convert_to_DMatrix(X_train, Y_train)
 
    xgb_model = train_model(train_data, valid_data, xgb_params, xgb_option)
    #auc = evaluate_model(xgb_model, valid_data)
    xgb_model.save_model('model/xgb_model.json')


    ## train NN model 
    xgb_model_path = 'model/xgb_model.json'
    method = 'tree-to-vector'

    assert method == 'tree-to-vector', 'tree-to-token'

    if method == 'tree-to-vector':
        xgbTree = TreeToVectorConverter(xgb_model_path, X_train.shape[1])
        transform_batch = transforms.Compose([TreeToVector(xgbTree)])
    elif method == 'tree-to-token':
        xgbTree = TreeToTokenConverter(xgb_model_path, X_train.shape[1])
        transform_batch = transforms.Compose([TreeToToken(xgbTree)])

    train_data = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            # transform 
            data = transform_batch(data)

