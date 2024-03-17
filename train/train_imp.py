import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import pruning
from train.metrics import accuracy_MNIST_CIFAR as accuracy
import pdb
import numpy as np
import dgl
from utils import GeneralizedCELoss


    
def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def train_model_and_masker(model_c, masker_c,  model_b, masker_b, optimizer_c, optimizer_b, sample_loss_ema_c, sample_loss_ema_b, device, data_loader, epoch, args):
    bias_criterion = GeneralizedCELoss(q=args.q)
    model_b.train()
    masker_b.train()
    model_c.train()
    masker_c.train()
    
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    mask_distribution = []
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):

        batch_graphs = dgl.batch(batch_graphs).to(device)

        batch_x = batch_graphs.ndata['feat'].to(device) 

        batch_labels = batch_labels.to(device)
        optimizer_c.zero_grad()
        optimizer_b.zero_grad()
        data_maskers = []

        batch_scores_c, data_mask_c = model_c(batch_graphs, batch_x, largest=True)
        batch_scores_b, data_mask_b = model_b(batch_graphs, batch_x, largest=False)

        norm_c = F.normalize(batch_scores_c, p=2, dim=1)
        norm_b = F.normalize(batch_scores_b, p=2, dim=1)
 
        labels = batch_labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device) - torch.eye(labels.shape[0]).to(device)

        inner_product1 = torch.exp(torch.mm(norm_c, norm_c.T) / 0.07)
        logits_max, _ = torch.max(inner_product1, dim=1, keepdim=True)
        inner_product1 = inner_product1 - logits_max.detach()
        inner_product2 = torch.exp(torch.mm(norm_c, norm_b.T) / 0.07)
        logits_max, _ = torch.max(inner_product2, dim=1, keepdim=True)
        inner_product2 = inner_product2 - logits_max.detach()

        sup_cl = 0
        uni_labs = torch.unique(batch_labels)
        for lab in uni_labs:
            sup_lab, sum = 0, 0
            idxs = torch.where(batch_labels==lab)[0]
            for idx in idxs:
                sum = torch.sum(mask[idx,:])
                sup_lab = sup_lab + torch.log(torch.sum(inner_product1[idx,:][mask[idx,:]==1]) / (torch.sum(inner_product1[idx,:]) + torch.sum(inner_product2[idx,:]) - inner_product1[idx,idx]))

            if sum != 0:
                sup_cl = sup_cl - sup_lab / sum
        
        mse = nn.MSELoss()
        mse_loss = mse(data_mask_c[0], data_mask_b[0])

        z_c = torch.cat((batch_scores_c, batch_scores_b.detach()), dim=1)
        z_b = torch.cat((batch_scores_c.detach(), batch_scores_b), dim=1)

        batch_scores_c1 = model_c.MLP_layer(z_c)
        batch_scores_b1 = model_b.MLP_layer(z_b)
        

        loss_dis_conflict = model_c.loss(batch_scores_c1, batch_labels).to(device)

        loss_dis_align = bias_criterion(batch_scores_b1, batch_labels)

        if epoch > args.swap_epochs:  
            loss_c = model_c.loss(batch_scores_c1, batch_labels).detach()
            loss_b = model_b.loss(batch_scores_b1, batch_labels).detach()

            loss_weight = loss_b / (loss_b + loss_c + 1e-8)

            indices = np.random.permutation(batch_scores_b.size(0))
            z_b_swap_0 = batch_scores_b[indices]       
            label_swap_0 = batch_labels[indices]

            mean = torch.mean(batch_scores_b, dim=1)
            var = torch.var(batch_scores_b, dim=1)

            indices = np.random.permutation(batch_scores_b.size(0))
            mean_swap = mean[indices]
            std_swap = var[indices]
            label_swap = batch_labels[indices]

            z_b_swap = (batch_scores_b - mean[:,None]) / var[:,None]
            z_b_swap = z_b_swap * std_swap[:,None] + mean_swap[:,None]

            z_mix_conflict = torch.cat((batch_scores_c, z_b_swap.detach()), dim=1)
            z_mix_conflict_0 = torch.cat((batch_scores_c, z_b_swap_0.detach()), dim=1)

            z_mix_align = torch.cat((batch_scores_c.detach(), z_b_swap), dim=1)
            z_mix_align_0 = torch.cat((batch_scores_c.detach(), z_b_swap_0), dim=1)

            pred_mix_conflict = model_c.MLP_layer(z_mix_conflict)
            pred_mix_conflict_0 = model_c.MLP_layer(z_mix_conflict_0)
            pred_mix_align = model_b.MLP_layer(z_mix_align)
            pred_mix_align_0 = model_b.MLP_layer(z_mix_align_0)

            loss_swap_conflict = (model_c.loss(pred_mix_conflict, batch_labels) * loss_weight.to(device) + model_c.loss(pred_mix_conflict_0, batch_labels) * loss_weight.to(device)) / 2     
            loss_swap_align = (bias_criterion(pred_mix_align, label_swap) + bias_criterion(pred_mix_align_0, label_swap_0)) / 2
            lambda1 = args.lambda1
        else:
            loss_swap_conflict = torch.tensor([0]).float()
            loss_swap_align = torch.tensor([0]).float()
            lambda1 = 0
           
        loss_swap = loss_swap_conflict.mean() + args.lambda_dis*loss_swap_align.mean()
        loss_dis = loss_dis_conflict.mean() + args.lambda_dis*loss_dis_align.mean()
        loss = loss_dis + lambda1 * loss_swap + args.lambda2 * sup_cl + args.lambda3 * mse_loss

        
        loss.backward()
        optimizer_b.step()
        optimizer_c.step()

        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores_c1, batch_labels)
        nb_data += batch_labels.size(0)

    mask_distribution = None
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer_c, mask_distribution

         

    

def eval_acc_with_mask(model_c, masker_c, model_b, device, data_loader, epoch, args, binary=False, val=False):

    model_c.eval()
    masker_c.eval()
    model_b.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = dgl.batch(batch_graphs).to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)

            batch_e = None
            batch_labels = batch_labels.to(device)
            batch_scores_c, _ = model_c(batch_graphs, batch_x, largest=True)
            batch_scores_b, _ = model_b(batch_graphs, batch_x, largest=False)


            scores_concat = torch.cat((batch_scores_c, batch_scores_b), dim=1)
            batch_scores = model_c.MLP_layer(scores_concat)
            loss = model_c.loss(batch_scores, batch_labels).mean()

            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc



def train_epoch(model, optimizer, device, data_loader, epoch, args):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_bias_labels) in enumerate(data_loader):
        batch_graphs = dgl.batch(batch_graphs).to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        #batch_e = batch_graphs.edata['feat'].to(device)
        batch_e = None
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    
        if iter % 40 == 0:
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                    'Epoch: [{}/{}]  Iter: [{}/{}]  Loss: [{:.4f}]'
                    .format(epoch + 1, args.eval_epochs, iter, len(data_loader), epoch_loss / (iter + 1), epoch_train_acc / nb_data * 100))
    
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer



def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_bias_labels) in enumerate(data_loader):
            batch_graphs = dgl.batch(batch_graphs).to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            #batch_e = batch_graphs.edata['feat'].to(device)
            batch_e = None
            batch_labels = batch_labels.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc
