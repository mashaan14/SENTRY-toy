import os

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

import params
from utils import make_variable


def train_src(feature_extractor_src, classifier, src_data_loader):

    # set train state for Dropout and BN layers
    feature_extractor_src.train()

    # setup criterion and optimizer
    optimizer_feature_extractor_src = optim.Adam(feature_extractor_src.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))

    len_data_loader = len(src_data_loader)

    for epoch in range(params.num_epochs_pre):

        if epoch + 1 <= params.num_epochs_pre:
            encoded_feat = np.zeros((len(src_data_loader), 2))
            label_pred = np.zeros((len(src_data_loader), 1), dtype=int)
            label_true = np.zeros((len(src_data_loader), 1), dtype=int)

        # zip source and target data pair
        for step, (samples_src, labels_src) in enumerate(src_data_loader):

            # prepare samples
            samples_src = make_variable(samples_src)

            # prepare source class label
            labels_src = make_variable(labels_src)

            # extract features
            feat_src = feature_extractor_src(samples_src)

            # predict source samples on classifier
            labels_src_pred = classifier(feat_src)

            # compute losses
            loss = F.cross_entropy(labels_src_pred, labels_src)

            # Backpropagation
            optimizer_feature_extractor_src.zero_grad()
            loss.backward()
            optimizer_feature_extractor_src.step()

            encoded_feat[step, :] = feature_extractor_src(samples_src).detach().numpy()
            label_pred[step, :] = classifier(feature_extractor_src(samples_src)).data.max(1)[1].detach().numpy()
            label_true[step, :] = labels_src.numpy()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "loss={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len_data_loader,
                              loss.item()))

        # plot visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Encoding features for source data')
        for g in np.unique(label_true):
            ix = np.where(label_true == g)
            ax1.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        for g in np.unique(label_pred):
            ix = np.where(label_pred == g)
            ax2.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        ax1.set_title('true labels')
        ax2.set_title('predicted labels')
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.savefig('plot-XS-train'+str(epoch)+'.png', bbox_inches='tight', dpi=600)

    return feature_extractor_src


def eval_src(encoder, classifier, data_loader, fig_title):
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    feat = np.zeros((len(data_loader), 2))
    label_pred = np.zeros((len(data_loader), 1), dtype=int)
    label_true = np.zeros((len(data_loader), 1), dtype=int)
    step = 0

    # evaluate network
    for (samples, labels) in data_loader:

        # make smaples and labels variable
        samples = make_variable(samples)
        labels = make_variable(labels)

        preds = classifier(encoder(samples))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

        feat[step, :] = samples.detach().numpy()
        label_pred[step, :] = preds.data.max(1)[1].detach().numpy()
        label_true[step, :] = labels.numpy()
        step += 1

    loss /= len(data_loader)
    acc = acc.item()/len(data_loader.dataset)
    ari = adjusted_rand_score(label_true.flatten(), label_pred.flatten())

    print("Avg Loss = {:.5f}, Avg Accuracy = {:2%}, ARI = {:.5f}".format(loss, acc, ari))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(fig_title)
    for g in np.unique(label_true):
        ix = np.where(label_true == g)
        ax1.scatter(feat[ix, 0], feat[ix, 1])

    for g in np.unique(label_pred):
        ix = np.where(label_pred == g)
        ax2.scatter(feat[ix, 0], feat[ix, 1])

    ax1.set_title('true labels')
    ax2.set_title('predicted labels')
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.savefig(fig_title+'.png', bbox_inches='tight', dpi=600)


def train_tgt(feature_extractor, classifier, src_data_loader, tgt_data_loader):
    """
    SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation (ICCV 2021)
    """

    # set train state for Dropout and BN layers
    feature_extractor.train()

    # setup criterion and optimizer
    optimizer_feature_extractor = optim.Adam(feature_extractor.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))


    for epoch in range(params.num_epochs):

        if epoch + 1 <= params.num_epochs:
            encoded_feat = np.zeros((len(tgt_data_loader), 2))
            label_pred = np.zeros((len(tgt_data_loader), 1), dtype=int)
            label_true = np.zeros((len(tgt_data_loader), 1), dtype=int)

        # prepare a queue to hold the predicted labels for the target samples
        queue = torch.zeros(params.queue_length)
        queue = make_variable(queue)
        pointer = 0

        # zip source and target data pair
        for step, ((samples_src, labels_src), (samples_tgt, labels_tgt, neighbors_tgt)) in enumerate(zip(src_data_loader, tgt_data_loader)):

            ###########################################################################################################
            # Compute loss_CE using source samples
            ###########################################################################################################

            # prepare source samples
            samples_src = make_variable(samples_src)

            # prepare source class label
            labels_src = make_variable(labels_src)

            # extract features
            feat_src = feature_extractor(samples_src)

            # predict source samples on classifier
            labels_src_pred = classifier(feat_src)

            # compute CE loss
            loss_CE = params.lambda_CE * F.cross_entropy(labels_src_pred, labels_src)

            ###########################################################################################################
            # Compute loss_IE using predicted labels on target samples
            ###########################################################################################################

            # prepare target samples
            samples_tgt = make_variable(samples_tgt)

            # extract features
            feat_tgt = feature_extractor(samples_tgt)

            # find the predicted class for the target sample
            labels_tgt_pred = feat_tgt.max(dim=1)[1].reshape(-1)

            # if the queue contains elements less than the queue, keep inserting
            if pointer < (queue.size()[0] - 1):
                queue[pointer] = labels_tgt_pred.detach()
                pointer += 1

            # else shift what's already in the queue one index to the left and place the current element at the last index
            else:
                queue = torch.roll(queue, -1, dims=0)
                queue[pointer] = labels_tgt_pred.detach()

            # count how many samples are classified to each class, the division is to get percentages
            queue_bincounts = torch.bincount(queue.long(), minlength=params.num_classes).float() / params.queue_length
            # take the log of these values, add epsilon to avoid undefined values
            queue_bincounts_log = torch.log(queue_bincounts + 1e-12).detach()

            # multiply the softmax of the current target sample with the class percentages.
            # this will exaggerate the predicted class of that target sample
            loss_IE = params.lambda_IE * torch.mean(torch.sum(feat_tgt.softmax(dim=1) * queue_bincounts_log.reshape(1, params.num_classes), dim=1))

            ###########################################################################################################
            # Compute loss_SENTRY using "Consistent" and "Inconsistent" target samples
            ###########################################################################################################

            # reshape the array holding neighbors features to the desired size
            neighbors_tgt = neighbors_tgt.reshape(params.num_neighbors, feat_tgt.size()[1])

            # prepare an array to hold the results of matching predicted labels
            neighbors_match_count = torch.zeros(params.num_neighbors)

            counter = 0
            for neighbors_tgt_curr in neighbors_tgt:
                # extract features
                neighbors_tgt_curr_feat = feature_extractor(neighbors_tgt_curr.unsqueeze(dim=0))

                # find the predicted class for the neighbor sample
                neighbors_tgt_curr_pred = neighbors_tgt_curr_feat.max(dim=1)[1].reshape(-1)

                # if the predicted class of the neighbor sample matches the predicted class of the current target sample:
                if labels_tgt_pred == neighbors_tgt_curr_pred:
                    # store the features of the neighbor sample as a positive sample
                    neighbors_tgt_pos_feat = neighbors_tgt_curr_feat
                    # insert 1 in the matching array
                    neighbors_match_count[counter] = 1

                # if the predicted class of the neighbor sample does not matche the predicted class of the current target sample:
                else:
                    # store the features of the neighbor sample as a negative sample
                    neighbors_tgt_neg_feat = neighbors_tgt_curr_feat
                    # insert 0 in the matching array
                    neighbors_match_count[counter] = 0

                counter += 1

            # if there are more than half of the neighbors match the predicted class of the current target sample:
            if torch.sum(neighbors_match_count) > np.ceil(params.num_neighbors / 2):
                # increase model confidence by minimizing predictive entropy
                loss_SENTRY = params.lambda_SENTRY * -torch.mean(torch.sum(neighbors_tgt_pos_feat.softmax(dim=1) *
                                                                    (torch.log(neighbors_tgt_pos_feat.softmax(dim=1) + 1e-12)), 1))
            else:
                # decrease model confidence by maximizing predictive entropy
                loss_SENTRY = params.lambda_SENTRY * torch.mean(torch.sum(neighbors_tgt_neg_feat.softmax(dim=1) *
                                                                   (torch.log(neighbors_tgt_neg_feat.softmax(dim=1) + 1e-12)), 1))

            loss = loss_CE + loss_IE + loss_SENTRY

            # Optimize
            optimizer_feature_extractor.zero_grad()
            loss.backward()
            optimizer_feature_extractor.step()


            encoded_feat[step, :] = feature_extractor(samples_tgt).detach().numpy()
            label_pred[step, :] = classifier(feature_extractor(samples_tgt)).data.max(1)[1].detach().numpy()
            label_true[step, :] = labels_tgt.numpy()

            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "loss_CE={:.4f} loss_IE={:.4f} loss_SENTRY={:.4f} loss={:.4f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_CE.item(),
                              loss_IE.item(),
                              loss_SENTRY.item(),
                              loss.item()))

        # plot visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Encoding features for target data')
        for g in np.unique(label_true):
            ix = np.where(label_true == g)
            ax1.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        for g in np.unique(label_pred):
            ix = np.where(label_pred == g)
            ax2.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        ax1.set_title('true labels')
        ax2.set_title('predicted labels')
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.savefig('plot-XT-train'+str(epoch)+'.png', bbox_inches='tight', dpi=600)

    return feature_extractor
