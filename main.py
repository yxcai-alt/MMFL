import json
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from Model.models import MMFL
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
    parser.add_argument('--dataset', type=str, default='TADPOLE', help='Dataset to use: TADPOLE or ABIDE')
    parser.add_argument('--task', type=str, default='AD_CN_SMCI', help='Task to perform')
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

    All_ACC_list = []
    All_AUC_list = []

    for splits_num, (train_mask, test_mask) in enumerate(Mask_list):

        model = MMFL(Adj_=Adj, out_channels=Class_num, feature_list=Feature_index_list, Device=device,
                       GCN_Model=GCN_Model, hidden=hidden_num, K=K, P=dropout_date, Dim_emb=Dim_emb).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        Loss_F = nn.CrossEntropyLoss(weight=label_weight).to(device)
        if CALR:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=Lr_Min)

        Best_ACC = 0.0
        Best_AUC = 0.0
        Best_model_state = None

        with tqdm(range(EPOCH), total=EPOCH) as pbar:
            for epoch in pbar:
                # train
                model.train()
                optimizer.zero_grad()
                out, Adj_ = model(Data_X)

                if torch.isnan(out).any() or torch.isnan(Adj_).any():
                    break

                loss = Loss_fn(Data_X, Adj_, out, Data_Y, train_mask, Loss_F, label_weight, device)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.8, norm_type=2)
                optimizer.step()

                _, train_labels_predict = out[train_mask].max(-1)
                train_acc = (Data_Y[train_mask] == train_labels_predict).sum().item()

                # eval
                model.eval()
                with torch.no_grad():
                    test_output, Adj_ = model(Data_X)
                    if torch.isnan(test_output).any() or torch.isnan(Adj_).any():
                        break

                _, test_labels_predict = test_output[test_mask].max(-1)
                test_acc = (Data_Y[test_mask] == test_labels_predict).sum().item()

                if CALR:
                    scheduler.step()

                Test_ACC = test_acc / test_num[splits_num]
                Train_ACC = train_acc / train_num[splits_num]
                Test_AUC = get_auc(y_true=F.one_hot(Data_Y[test_mask].cpu(), num_classes=Class_num),
                                y_pred=test_output[test_mask].cpu())

                if Test_ACC > Best_ACC:
                    Best_ACC = Test_ACC

                    if Test_AUC > Best_AUC:
                        Best_AUC = Test_AUC

                        if Save_Model:
                            Best_model_state = model.state_dict()

                pbar.set_description('Split：{} Epoch: {} Train Loss: {:.4f} Train ACC: {:.4f} Test ACC: {:.4f}'.format(
                    splits_num + 1, epoch + 1, loss.item(), Train_ACC, Test_ACC))
        
        All_ACC_list.append(Best_ACC)
        All_AUC_list.append(Best_AUC)
                
        print(f'Split {splits_num + 1} Best ACC: {Best_ACC} Best AUC: {Best_AUC}')


        # if Save_Model:
        #     torch.save(Best_model_state,
        #                os.path.join(Root_path, 'RESULT/Model_Weights/{}_{}_Model_Splits_{}.pth'.format(DATA_SET, Task,
        #                                                                                                splits_num)))

    ACC_mean = np.mean(All_ACC_list)
    AUC_mean = np.mean(All_AUC_list)

    ACC_std = np.std(All_ACC_list)
    AUC_std = np.std(All_AUC_list)

    print(f'ACC: Mean: {ACC_mean} Std: {ACC_std}')
    print(f'AUC: Mean: {AUC_mean} Std: {AUC_std}')

    result = {'ACC': {'Mean': ACC_mean, 'Std': ACC_std}, 'AUC': {'Mean': AUC_mean, 'Std': AUC_std}}
    
    pid = os.getpid()
    random.seed(pid)
    letters = string.ascii_letters + string.digits
    str_ =''.join(random.choice(letters) for i in range(10))

    Save_History_path =os.path.join(Root_path, f'RESULT/{DATA_SET}/{Task}/{str_}.json')
    os.makedirs(f'RESULT/{DATA_SET}/{Task}/', exist_ok=True)

    json.dump(result, open(Save_History_path, 'w'))



