import os
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def construct_symmetric_matrix(original_matrix):

    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")

    return result_matrix



def feature_to_adj(feature, k_nearest_neighobrs=7, rm_common_neighbors=2):
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(feature)
    adj_wave = nbrs.kneighbors_graph(feature)

    np_adj_wave = construct_symmetric_matrix(adj_wave.toarray())
    adj = sp.csc_matrix(np_adj_wave)

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj = adj.toarray() 

    b = np.nonzero(adj)
    rows = b[0]
    cols = b[1]
    dic = {}
    for row, col in zip(rows, cols):
        if row in dic.keys():
            dic[row].append(col)
        else:
            dic[row] = []
            dic[row].append(col)
    for row, col in zip(rows, cols):
        if len(set(dic[row]) & set(dic[col])) < rm_common_neighbors:
            adj[row][col] = 0
    adj = sp.csc_matrix(adj)
    adj.eliminate_zeros()

    adj_w = adj.toarray()  + sp.eye(adj.shape[0])

    adj_hat = adj_w

    return adj_hat


def Get_Mask(Data_X, Data_Y, Device, n_splits=10, random_seed=0):
    Mask_list = []
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
    for train_index, test_index in skf.split(Data_X, Data_Y):
        train_mask = torch.zeros(Data_X.shape[0], dtype=torch.bool)
        train_mask[train_index] = True
        test_mask = ~train_mask
        Mask_list.append([train_mask.to(Device), test_mask.to(Device)])
    return Mask_list


def count_elements(matrix):
    unique_elements, counts = np.unique(matrix, return_counts=True)
    result = dict(zip(unique_elements, counts))
    sorted_dict = dict(sorted(result.items()))
    value_list = list(sorted_dict.values())
    label_weight = torch.from_numpy(np.array(value_list) / matrix.shape[0])

    return label_weight


def get_data(Feature_Data_path, Feature_dict_path, Device=None, k_nearest_neighobrs=7, rm_common_neighbors=2,
             Random_seed=0):
    if not Device:
        Device = torch.device('cpu')

    Feature_dict = np.load(Feature_dict_path, allow_pickle=True).item()
    Feature_Data = pd.read_csv(Feature_Data_path, low_memory=False)

    Feature_Data = Feature_Data.sample(frac=1)
    Data_X = Feature_Data.iloc[:, :-1]
    Data_Y = Feature_Data.iloc[:, -1] - 1

    Feature_Data_index = Feature_Data.index.tolist()

    Data_Adj_list = []
    for key, value in Feature_dict.items():
        Adj_ = feature_to_adj(Data_X[value].values, k_nearest_neighobrs=k_nearest_neighobrs,
                              rm_common_neighbors=rm_common_neighbors)
        Data_Adj_list.append(torch.from_numpy(Adj_))
    Adj = torch.stack(Data_Adj_list)

    Data_X = torch.from_numpy(Data_X.values).float()
    Data_Y = torch.from_numpy(Data_Y.values).long()

    Mask_list = Get_Mask(Data_X, Data_Y, Device, n_splits=10, random_seed=Random_seed)

    Data_X = Data_X.to(Device)
    Data_Y = Data_Y.to(Device)
    Adj = Adj.float()
    Adj = Adj.to(Device)

    in_channels = Data_X.shape[1]
    Class_num = len(torch.unique(Data_Y))

    train_num = [Mask_[0].int().sum().item() for Mask_ in Mask_list]
    test_num = [Mask_[1].int().sum().item() for Mask_ in Mask_list]

    Col_to_index_dict = {col: idx for idx, col in enumerate(Feature_Data.columns)}
    Feature_index_list = []
    for key, value in Feature_dict.items():
        Feature_index_list.append([Col_to_index_dict[col] for col in value])

    label_weight = count_elements(Data_Y.cpu().numpy())
    label_weight = label_weight.float()

    return Data_X, Data_Y, Adj, Mask_list, in_channels, Class_num, train_num, test_num, Feature_index_list, label_weight.to(
        Device), Feature_Data_index



def get_auc(y_true, y_pred):
    y_pred = y_pred.detach().numpy()
    y_true = y_true.detach().numpy()
    auc_scores = roc_auc_score(y_true, y_pred)

    return auc_scores


def get_confusion_matrix(AUC_Index, ALL_label_predict, All_label_truth):
    ALL_confusion_matrix = []
    for index_, label_predict_, All_label_truth_ in zip(AUC_Index, ALL_label_predict, All_label_truth):
        ALL_confusion_matrix.append(confusion_matrix(y_true=All_label_truth_[index_].detach().numpy(),
                                                     y_pred=label_predict_[index_].max(-1)[1].detach().numpy()))

    return ALL_confusion_matrix


def Calculate_indicators(ALL_acc_test, ALL_label_predict, All_label_truth):
    max_val = ALL_acc_test.max(-1)[0].unsqueeze(-1)
    indices = (ALL_acc_test == max_val).nonzero()
    groups = defaultdict(list)
    for row in indices:
        row = row.tolist()
        key = row[0]
        groups[key].append(row[1:])
    Dict_indices = {k: torch.tensor(v).squeeze(-1) for k, v in groups.items()}
    ALL_AUC = []
    AUC_Index = []
    for Split_, Index_l in Dict_indices.items():
        AUC_List = []
        for Index_ in Index_l:
            Predict_ = ALL_label_predict[Split_][Index_]
            Label_ = F.one_hot(All_label_truth[Split_][Index_], num_classes=Predict_.shape[-1])
            AUC_List.append(get_auc(y_true=Label_, y_pred=Predict_))
        AUC_Index_ = np.argmax(AUC_List)
        AUC_Index.append(Index_l[AUC_Index_])
        ALL_AUC.append(np.max(AUC_List))

    ALL_confusion_matrix = get_confusion_matrix(AUC_Index, ALL_label_predict, All_label_truth)

    return ALL_AUC, ALL_confusion_matrix
