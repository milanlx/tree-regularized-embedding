import torch 
import xgboost as xgb


"""Tree to Vector
"""
class TreeToVector: 
    def __init__(self, xgbTree, dtype=torch.float): 
        self.xgbTree = xgbTree 
        self.dtype = dtype
        
    def __call__(self, tensor): 
        output = self.tree_encoder(tensor)
        return output 
    
    def tree_encoder(self, tensor): 
        # fill nan with -1 
        tensor = torch.nan_to_num(tensor, nan=-1.0)
        output = self.postprocessing(
                     tensor, 
                     self.xgbTree.multiply_matrix, 
                     self.xgbTree.offset_vector)
        return output 
        
    def postprocessing(self, x, multiply_matrix, offset_vector): 
        device = x.device
        multiply_matrix = multiply_matrix.to(device)
        offset_vector = offset_vector.to(device)
        
        x = torch.matmul(x, multiply_matrix)
        x -= offset_vector
        x[x > 0] = 1.0
        x[x < 0] = 0.0
        return x
    
    
class TreeToVectorConverter:
    def __init__(self, xgb_model_path, num_variable): 
        self.df_trees = self.load_xgb_model(xgb_model_path)
        self.num_variable = num_variable
        self.num_tree = self.df_trees['Tree'].nunique()
        self.tree_dict = self.get_tree_dict()
        self.num_encode = self.get_num_encode()
        self.multiply_matrix, self.offset_vector = self.get_encode_element()
        
    def load_xgb_model(self, model_path): 
        xgb_model = xgb.Booster()
        xgb_model.load_model(model_path)
        df_trees = xgb_model.trees_to_dataframe()
        return df_trees
    
    def get_tree_dict(self): 
        tree_list = extract_trees(self.df_trees)
        tree_dict = self.dedup_tree(tree_list)
        return tree_dict 
    
    # get compact tree with duplicated threshold removed (given precision)
    def dedup_tree(self, trees, precision=2): 
        compact_trees = {}
        for tree in trees: 
            for variable in tree.keys(): 
                split_val = tree[variable]
                # only keep the first x precision
                split_val = [round(item, precision) for item in split_val]
                # add to global dict 
                if variable not in compact_trees: 
                    compact_trees[variable] = split_val
                else:
                    compact_trees[variable] = compact_trees[variable] + split_val
        # remove duplicates 
        for key in compact_trees: 
            compact_trees[key] = list(set(compact_trees[key]))
        return compact_trees
    
    def get_num_encode(self): 
        num_encode = 0 
        for key in self.tree_dict: 
            num_encode += len(self.tree_dict[key])
        return num_encode 
    
    def get_encode_element(self): 
        m, d = self.num_variable, self.num_encode
        
        multiply_matrix = torch.zeros((m, d))
        offset_vector = torch.zeros((1, d))
        
        col_idx = 0
        for feature in self.tree_dict: 
            # row idx 
            idx = int(feature[1:])
            for split in self.tree_dict[feature]:
                multiply_matrix[idx, col_idx] = 1.0
                offset_vector[0, col_idx] = split
                col_idx += 1
        return multiply_matrix, offset_vector

    
"""Tree to Token
"""
class TreeToToken: 
    def __init__(self, xgbTree, dtype=torch.float): 
        self.xgbTree = xgbTree 
        self.dtype = dtype
    
    def __call__(self, tensor): 
        num_data = tensor.size()[0]
        num_tree = self.xgbTree.num_tree
        num_encode = self.xgbTree.num_encode
        output = torch.zeros((num_data, num_tree, num_encode), dtype=self.dtype)
        output = self.tree_encoder(tensor, output)
        return output 
    
    def tree_encoder(self, tensor, output): 
        # fill nan with -1 
        tensor = torch.nan_to_num(tensor, nan=-1.0)
        for i in range(self.xgbTree.num_tree): 
            multiply_matrix = self.xgbTree.multiply_list[i]
            offset_vector   = self.xgbTree.offset_list[i]
            padding_vector  = self.xgbTree.padding_list[i]
            x_encode = self.postprocessing(tensor, multiply_matrix, offset_vector, padding_vector)
            output[:, i, :] = x_encode 
        return output 
    
    def postprocessing(self, x, multiply_matrix, offset_vector, padding_vector): 
        device = x.device
        multiply_matrix = multiply_matrix.to(device)
        offset_vector = offset_vector.to(device)
        padding_vector = padding_vector.to(device)
        
        x = torch.matmul(x, multiply_matrix)
        x -= offset_vector
        x[x > 0] = 1.0
        x[x < 0] = 0.0
        x += padding_vector
        return x 

    
class TreeToTokenConverter:
    def __init__(self, xgb_model_path, num_variable): 
        self.df_trees = self.load_xgb_model(xgb_model_path)
        self.num_variable = num_variable
        self.num_tree = self.df_trees['Tree'].nunique()
        self.num_encode = self.get_num_encode()
        self.tree_list = self.get_tree_list()
        self.multiply_list, self.offset_list, self.padding_list = self.get_encode_element()
    
    def load_xgb_model(self, model_path): 
        xgb_model = xgb.Booster()
        xgb_model.load_model(model_path)
        df_trees = xgb_model.trees_to_dataframe()
        return df_trees
    
    def get_tree_list(self): 
        tree_list = []
        for i in range(self.num_tree): 
            df_tree = self.df_trees.loc[self.df_trees['Tree']==i]
            df_tree.reset_index(inplace=True)
            tree_list.append(df_tree)
        return tree_list    
    
    def get_num_encode(self): 
        num_encode = 0 
        for i in range(self.num_tree):
            df_tree = self.df_trees.loc[self.df_trees['Tree']==i]
            num_encode = max(num_encode, len(df_tree))
        return num_encode 

    def get_encode_element(self): 
        multiply_list, offset_list, padding_list = [], [], []
        for tree in self.tree_list: 
            multiply_matrix, offset_vector, padding_vector = self.one_tree_encoder(tree)
            multiply_list.append(multiply_matrix)
            offset_list.append(offset_vector)
            padding_list.append(padding_vector)
        return multiply_list, offset_list, padding_list
    
    def one_tree_encoder(self, tree): 
        m, d = self.num_variable, self.num_encode
        
        multiply_matrix = torch.zeros((m, d))
        offset_vector = torch.zeros((1, d))
        padding_vector = torch.zeros((1, d))
        
        for i in range(len(tree)):
            feature = tree['Feature'][i]
            # update multiply and offset
            if feature != 'Leaf': 
                idx = int(tree['Feature'][i][1:])
                split = float(tree['Split'][i])
                multiply_matrix[idx, i] = 1.0
                offset_vector[0, i] = split
            # update padding vector 
            elif feature == 'Leaf': 
                padding_vector[0, i] = 0.5 
        # update padding vector 
        for i in range(len(tree), d): 
            padding_vector[0, i] = -1 
        return multiply_matrix, offset_vector, padding_vector 
    
    
"""helper function 
"""
def extract_trees(xgb_model):
    """ extract trees from XGB for binary classification cases
    Input: XGB model 
    Output: list of dict (tree)
    """
    trees = []
    #df = xgb_model._Booster.trees_to_dataframe()
    #df = xgb_model.trees_to_dataframe()
    df = xgb_model
    num_tree = df['Tree'].nunique()
    for i in range(num_tree): 
        tree_temp = {}
        mask = (df['Tree']==i) & (df['Feature']!='Leaf')
        df_temp = df.loc[mask, ['Feature', 'Split']] 
        for _, row in df_temp.iterrows(): 
            feature, split = row['Feature'], row['Split']
            if feature not in tree_temp:
                tree_temp[feature] = [split]
            else:
                tree_temp[feature].append(split)
        trees.append(tree_temp)
    return trees 