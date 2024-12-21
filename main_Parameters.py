import json
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from Model.models import *
from Loss.loss_fn import Loss_fn
from utils import get_data, get_auc
import argparse
import string


def SET_Random(seed):
    """Set the seed for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    '''
     TADPOLE

     SMCI_PMCI 
     AD_CN_SMCI 
     AD_CN_SMCI_PMCI

     ADNI

     AD_CN
     SMC_EMCI_LMCI
     AD_CN_EMCI_LMCI

     ABIDE ABIDE-5

     NC_ASD   python main.py --dataset ABIDE --task NC_ASD
     '''

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--dataset', type=str, default='ADNI', help='Dataset to use: TADPOLE or ABIDE')
    parser.add_argument('--task', type=str, default='SMC_EMCI_LMCI', help='Task to perform')
    parser.add_argument('--knn', type=int, default=7, help='Number of nearest neighbors')

    args = parser.parse_args()

    DATA_SET = args.dataset
    Task = args.task
    k_nearest_neighobrs = args.knn

    print('Dataset:', DATA_SET, 'Task:', Task, 'KNN:', k_nearest_neighobrs)


    Save_Model = False
    CALR = True
    weight_decay = 0.0009  
    EPOCH = 500  

    Shuffle = True
    n_splits = 10
    Random_seed = 0

    if CALR:
        lr = 0.01
        Lr_Min = 0.0001
    else:
        lr = 0.001 
    hidden_num = 32
    Dim_emb = 32
    dropout_date = 0.6
    K = 1
    GCN_Model = 'Cheby_GCN'  

    SET_Random(Random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    Root_path = os.path.dirname(os.path.abspath(__file__))

    if DATA_SET == 'TADPOLE':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, 'DATASET/TADPOLE/{}_ADNI_processed_standard_data.csv'.format(Task))
        Feature_dict_path = os.path.join(Root_path, 'DATASET/TADPOLE/{}_ADNI_modal_feat_dict.npy'.format(Task))
    
    elif DATA_SET == 'ADNI':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, 'DATASET/ADNI/Select_NEW_{}__ADNI_data.csv'.format(Task))
        Feature_dict_path = os.path.join(Root_path, 'DATASET/ADNI/Select_NEW_{}__ADNI_modal_feat_dict.npy'.format(Task))

    elif DATA_SET == 'ABIDE':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, 'DATASET/ABIDE/ABIDE_processed_data_modal.csv')
        Feature_dict_path = os.path.join(Root_path, 'DATASET/ABIDE/ABIDE_modal_feat_dict.npy')

    elif DATA_SET == 'ABIDE-5':
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, 'DATASET/ABIDE/ABIDE_processed_data_modal.csv')
        Feature_dict_path = os.path.join(Root_path, 'DATASET/ABIDE/ABIDE_modal_feat_dict.npy')

    else:
        print(f'Choose {DATA_SET} Dataset Task {Task}')
        Feature_Data_path = os.path.join(Root_path, 'DATASET/TADPOLE/{}_ADNI_processed_standard_data.csv'.format(Task))
        Feature_dict_path = os.path.join(Root_path, 'DATASET/TADPOLE/{}_ADNI_modal_feat_dict.npy'.format(Task))
    

    Data_X, Data_Y, Adj, Mask_list, in_channels, Class_num, train_num, test_num, Feature_index_list, label_weight, Feature_Data_index = get_data(
        Feature_Data_path,
        Feature_dict_path,
        Device=device,
        k_nearest_neighobrs=k_nearest_neighobrs,
        rm_common_neighbors=2,
        Random_seed=Random_seed)
    
    M_SSNs_Encoder = MultimodalStateSpaceNetworks(Feature_index_list, device, dim_embedding=Dim_emb).to(device)

    M_SSNs_Encoder_p = nn.Sequential(nn.LayerNorm(Dim_emb),
                                    nn.Dropout(p=dropout_date),
                                    nn.LeakyReLU(),
                                    ).to(device)
    M_SSNs_Decoder = nn.Sequential(nn.Linear(in_features=Dim_emb * len(Feature_index_list), out_features=Dim_emb),
                                nn.LayerNorm(Dim_emb),
                                nn.LeakyReLU()
                                ).to(device)

    M_GSL = MultimodalGraphStructureLearning(Adj, Feature_index_list, device, Dim_emb, rate=0).to(device)
    M_GF  = MultimodalGraphFusion(Adj.shape, device).to(device)


    if GCN_Model == 'GCN':
        GCN = GCN(Dim_emb=Dim_emb, hidden=hidden_num, out_channels=Class_num, P=dropout_date)
    elif GCN_Model == 'Cheby_GCN':
        GCN = Cheb_GCN(Dim_emb=Dim_emb, hidden=hidden_num, out_channels=Class_num, P=dropout_date, K=K)
    
    M_SSNs_params = sum(p.numel() for p in M_SSNs_Encoder.parameters())
    M_SSNs_p_params = sum(p.numel() for p in M_SSNs_Encoder_p.parameters())
    M_SSNs_Decoder_params = sum(p.numel() for p in M_SSNs_Decoder.parameters())
    n = Adj.shape[0]
    M_GSL_params = sum(p.numel() for p in M_GSL.parameters()) - ((n * n - n) // 2)*len(Feature_index_list)
    M_GF_params = sum(p.numel() for p in M_GF.parameters())
    GCN_params = sum(p.numel() for p in GCN.parameters())

    M_SSNs_params =  M_SSNs_params + M_SSNs_p_params + M_SSNs_Decoder_params
    print('M_SSNs_params:', M_SSNs_params)
    print('M_GSL_params:', M_GSL_params)
    print('M_GF_params:', M_GF_params)
    print('GCN_params:', GCN_params)
    print('Total:', M_SSNs_params + M_GSL_params + M_GF_params + GCN_params)




   