from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import numpy as np
import matplotlib.pyplot as plt
import torch

import params
import models
import utils
import core


def rotate(xy, theta):
    """
    Returns a rotated set of points.
    """
    s = np.sin(theta * np.pi / 180)
    c = np.cos(theta * np.pi / 180)
    center_of_rotation = np.mean(xy, axis=0)

    xyr = np.zeros((xy.shape[0], xy.shape[1]))
    xyr[:, 0] = (c * (xy[:, 0]-center_of_rotation[0])) - (s * (xy[:, 1]-center_of_rotation[1])) + center_of_rotation[0]
    xyr[:, 1] = (s * (xy[:, 0]-center_of_rotation[0])) + (c * (xy[:, 1]-center_of_rotation[1])) + center_of_rotation[1]

    return xyr

XS = np.genfromtxt('data/dbmoon1000_Instances.csv', delimiter=",")
YS = np.genfromtxt('data/dbmoon1000_Labels.csv', delimiter=",")

XT = rotate(XS, 30)
YT = YS

XT = XT[399:-1, :]
YT = YT[399:-1]

XS_train, XS_test, YS_train, YS_test = train_test_split(XS, YS, test_size=0.2, random_state=42)
XT_train, XT_test, YT_train, YT_test = train_test_split(XT, YT, test_size=0.2, random_state=42)

# n_neighbors=k means k-1 neighbors because the first index is the point itself
neigh = NearestNeighbors(n_neighbors=params.num_neighbors+1)
neigh.fit(XT_train)
_, XT_train_neighbors_idx = neigh.kneighbors(XT_train)
XT_train_neighbors_idx = XT_train_neighbors_idx[:, 1:]
XT_train_neighbors = XT_train[XT_train_neighbors_idx]

fig = plt.figure()  # figsize=(6, 6)
ax = fig.add_subplot(111)
for g in np.unique(YS_train):
    ix = np.where(YS_train == g)
    ax.scatter(XS_train[ix, 0], XS_train[ix, 1], label='source class '+str(int(g)))

ax.scatter(XT[:, 0], XT[:, 1], c='k', label='target')

plt.legend(loc=0)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
plt.savefig('dataset.png', bbox_inches='tight', dpi=600)


XS_train_tensor = torch.Tensor(XS_train)
XS_test_tensor = torch.Tensor(XS_test)
YS_train_tensor = torch.LongTensor(YS_train)
YS_test_tensor = torch.LongTensor(YS_test)

XT_train_tensor = torch.Tensor(XT_train)
XT_train_neighbors_tensor = torch.Tensor(XT_train_neighbors)
XT_test_tensor = torch.Tensor(XT_test)
YT_train_tensor = torch.LongTensor(YT_train)
YT_test_tensor = torch.LongTensor(YT_test)

XS_train_dataset = torch.utils.data.TensorDataset(XS_train_tensor, YS_train_tensor)
XS_test_dataset = torch.utils.data.TensorDataset(XS_test_tensor, YS_test_tensor)

XT_train_dataset = torch.utils.data.TensorDataset(XT_train_tensor, YT_train_tensor, XT_train_neighbors_tensor)
XT_test_dataset = torch.utils.data.TensorDataset(XT_test_tensor, YT_test_tensor)

XS_train_dataloader = torch.utils.data.DataLoader(XS_train_dataset)
XS_test_dataloader = torch.utils.data.DataLoader(XS_test_dataset)

XT_train_dataloader = torch.utils.data.DataLoader(XT_train_dataset)
XT_test_dataloader = torch.utils.data.DataLoader(XT_test_dataset)

feature_extractor_src = models.Encoder()
feature_extractor_src.apply(utils.init_weights)

cls = models.Classifier()
cls.apply(utils.init_weights)

feature_extractor_src = core.train_src(feature_extractor_src, cls, XS_train_dataloader)

""" eval source encoder on test set from source/target datasets """
print(">>> Testing source data using source feature extractor <<<")
core.eval_src(feature_extractor_src, cls, XS_test_dataloader, fig_title='Testing source data using source feature extractor')
print(">>> Testing target data using source feature extractor <<<")
core.eval_src(feature_extractor_src, cls, XT_test_dataloader, fig_title='Testing target data using source feature extractor')

feature_extractor = models.Encoder()
feature_extractor.apply(utils.init_weights)

feature_extractor = core.train_tgt(feature_extractor, cls, XS_train_dataloader, XT_train_dataloader)

""" eval target encoder on test set of target dataset """
print(">>> Testing target data using the feature extractor <<<")
core.eval_src(feature_extractor, cls, XT_test_dataloader, fig_title='Testing target data using target feature extractor')
#
