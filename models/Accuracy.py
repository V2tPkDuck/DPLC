import random
import torch
import numpy as np
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score, accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

import torch_clustering

def cluster_accuracy(y_true, y_pre,verbose=True):
    y_best = best_match(y_true, y_pre)
    # for c in np.unique(y_true):
    #     print([c,np.sum(y_true==c),np.sum(y_best==c)])
    # # calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = normalized_mutual_info_score(y_true, y_pre)
    ari = adjusted_rand_score(y_true, y_pre)
    ca = class_acc(y_true, y_best)

    return acc,nmi, ari,ca


def best_match(y_true, y_pre):
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]
    # print(c)
    return y_best


def class_acc(y_true, y_pre):
    """
    calculate each class's acc
    :param y_true:
    :param y_pre:
    :return:
    """
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
        y_c_p = y_pre[np.nonzero(y_true == c)]
        accuracy = accuracy_score(y_c, y_c_p)
        ca.append(accuracy)
        # print([c,len(y_c),accuracy])
    ca = np.array(ca)
    return ca
def clustering(features, n_clusters,distributed=True,random_state=0):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
        if torch.cuda.is_available():
            features = features.cuda()
    kwargs = {
        'metric': 'cosine' ,#if self.l2_normalize else 'euclidean',
        'distributed': distributed, #True
        'random_state': random_state,
        'n_clusters': n_clusters,
        'verbose': False
    }
    clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

    psedo_labels = clustering_model.fit_predict(features)
    cluster_centers = clustering_model.cluster_centers_
    return psedo_labels, cluster_centers