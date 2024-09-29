# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:37:53 2021
@author: Hristo Petkov
"""

"""
Modifications copyright (C) 2021 Hristo Petkov
This file used to be called train.py in the original DAG-GNN code.
Modifications are as follows:
  -Addition of a new discriminator class
  -Addition of a new training disctance (i.e the wasserstein distance in the form of adversarial loss)
"""

"""
@inproceedings{yu2019dag,
  title={DAG-GNN: DAG Structure Learning with Graph Neural Networks},
  author={Yue Yu, Jie Chen, Tian Gao, and Mo Yu},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  year={2019}
}
@inproceedings{xu2019modeling,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
"""

import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import pandas as pd
import os
import pickle

from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import Linear, Sequential, LeakyReLU, Dropout, BatchNorm1d
from Utils import preprocess_adj_new, preprocess_adj_new1
from Utils import nll_gaussian, kl_gaussian_sem,  nll_catogrical
from Utils import _h_A
from Utils import count_accuracy

class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_xdims, n_hid, n_out, adj_A, device, data_type):
        super(MLPEncoder, self).__init__()
        
        self.data_type = data_type
        
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        
        if self.data_type == 'benchmark':
            self.Wa = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)

        self.fc1 = Linear(n_xdims, n_hid, bias = True)
        self.fc2 = Linear(n_hid, n_out, bias = True) 
        
        self.device = device
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
   

    def forward(self, inputs):
        
        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)

        adj_Aforz = preprocess_adj_new(adj_A1, self.device)
        
        if self.data_type == 'benchmark':
        
            H1 = F.leaky_relu((self.fc1(inputs)))
        else:
            H1 = F.relu((self.fc1(inputs))) 
            
        x = (self.fc2(H1))
        
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa #mu
        
        return x, logits, adj_A1, self.Wa

class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_z, n_out, n_hid, device, data_type):
        super(MLPDecoder, self).__init__()
        
        self.out_fc1 = Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = Linear(n_hid, n_out, bias = True)
        
        self.device = device
        self.data_type = data_type

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A, self.device)
        
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa
        
        if self.data_type == 'benchmark':
            H3 = F.leaky_relu(self.out_fc1((mat_z)))
        else:
            H3 = F.relu(self.out_fc1((mat_z)))
            
        out = self.out_fc2(H3)

        return mat_z, out    
    
class Discriminator(nn.Module):
    """Discriminator module."""
    def __init__(self, input_dim, discriminator_dim, negative_slope, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        self.negative_slope = negative_slope
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(self.negative_slope), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def calc_gradient_penalty(self, real_data, fake_data, data_type, device='cpu', pac=10, lambda_=10):
        
        # reshape data
        real_data = real_data.squeeze()
        fake_data = fake_data.squeeze()
        
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))
    
class AAE_WGAN_GP(nn.Module):
    """DAG-AAE model/framework."""
    def __init__(self, args, adj_A):
        super(AAE_WGAN_GP, self).__init__()
        
        self.data_type = args.data_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = args.batch_size
        
        self.discriminator_steps = args.discriminator_steps
        self.epochs = args.epochs
        self.lr = args.lr
        
        self.c_A = args.c_A
        self.lambda_A = args.lambda_A
        self.tau_A = args.tau_A
        self.graph_threshold = args.graph_threshold
        
        self.x_dims = args.x_dims
        self.z_dims = args.z_dims
        self.encoder_hidden = args.encoder_hidden
        self.decoder_hidden = args.decoder_hidden
        self.adj_A = adj_A
        
        self.k_max_iter = int(args.k_max_iter)
        self.h_tol = args.h_tol
        
        self.h_A_new = torch.tensor(1.)
        self.h_A_old = np.inf
        
        self.discrete_columns = args.discrete_column_names_list
        self.data_variable_size = self.adj_A.shape[1]
        
        self.lr_decay = args.lr_decay
        self.gamma = args.gamma
        self.negative_slope = args.negative_slope
        
        self.save_directory = args.save_directory
        self.load_directory = args.load_directory
        
    def forward(self, inputs):
        en_outputs, logits, new_adjA, Wa = self.encoder(inputs)
        mat_z, de_outputs = self.decoder(logits, new_adjA, Wa)
        return en_outputs, logits, new_adjA, mat_z, de_outputs
                
    def update_optimizer(self, optimizer, original_lr, c_A):
        '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group['lr'] = lr

        return optimizer, lr
        
    def vae_loss(self, data, de_outputs, logits, origin_A, tau_A, data_variable_size, lambda_A, c_A, device):
        
        target = data
        preds = de_outputs
        variance = 0.
        
        # reconstruction accuracy loss
        if self.data_type == 'benchmark':
            loss_nll = nll_catogrical(preds, target)
        else:   
            loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = origin_A 
        sparse_loss = tau_A * torch.sum(torch.abs(one_adj_A))

        # compute h(A)
        h_A = _h_A(origin_A, data_variable_size, device)
        
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss
        
        return loss, preds, target, loss_nll, loss_kl 
    
    def train(self, train_loader, epoch, best_val_loss, best_epoch, best_mse_loss, best_mse_data,
                        best_shd, best_shd_graph, ground_truth_G, lambda_A, c_A, optimizerV, optimizerD):
        '''training algorithm for a single epoch'''
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        self.schedulerV.step()
        self.schedulerD.step()

        # update optimizer
        optimizerV, lr = self.update_optimizer(optimizerV, self.lr, c_A)
        optimizerD, lr = self.update_optimizer(optimizerD, self.lr, c_A)

        for batch_idx, (data, relations) in enumerate(train_loader):
            for n in range(self.discriminator_steps):
                ###################################################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###################################################################
                
                data, relations = Variable(data.to(self.device)).double(), Variable(relations.to(self.device)).double()
                
                if self.data_type != 'synthetic':
                    data = data.unsqueeze(2)
                
                optimizerD.zero_grad()
                
                en_outputs, logits, origin_A, mat_z, de_outputs = self(data)
                
                y_fake = self.discriminator(de_outputs)
            
                y_real = self.discriminator(data)
                
                if self.x_dims > 1:
                    #vector case
                    pen = self.discriminator.calc_gradient_penalty(
                        data.view(-1, data.size(1) * data.size(2)), de_outputs.view(-1, de_outputs.size(1) * de_outputs.size(2)), self.data_type, self.device) 
                    loss_d = -(torch.mean(F.softplus(y_real)) - torch.mean(F.softplus(y_fake)))
                else:
                    #normal continious and discrete data case
                    pen = self.discriminator.calc_gradient_penalty(
                            data, de_outputs, self.data_type, self.device) 
                    loss_d = -(torch.mean(F.softplus(y_real)) - torch.mean(F.softplus(y_fake)))
                    
                pen.backward(retain_graph=True)
                loss_d.backward()
                loss_d = optimizerD.step() 
            
            ###################################################
            # (2) Update G network which is the decoder of VAE
            ###################################################
            
            data, relations = Variable(data.to(self.device)).double(), Variable(relations.to(self.device)).double()
            
            optimizerV.zero_grad()
            
            en_outputs, logits, origin_A, mat_z, de_outputs = self(data)
            
            loss, preds, target, loss_nll, loss_kl = self.vae_loss(data, de_outputs, logits, origin_A, self.tau_A, self.data_variable_size, lambda_A, c_A, self.device)

            loss.backward(retain_graph=True)
            loss = optimizerV.step()
            
            ###############################################
            # (3) Update G network: maximize log(D(G(z)))
            ###############################################

            optimizerV.zero_grad()
            
            y_fake = self.discriminator(de_outputs.data.clone()) #cloning is absolutely necessary here
            
            lossVAE = -torch.mean(y_fake)
            lossVAE.backward()
            lossVAE = optimizerV.step() 
            
            # compute metrics
            graph = origin_A.data.clone().cpu().numpy()
            graph[np.abs(graph) < self.graph_threshold] = 0
            
            mse = F.mse_loss(preds, target).item()
            
            if ground_truth_G != None:
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
                shd_trian.append(shd)
                
                if best_shd == np.inf and best_mse_loss == np.inf:
                    best_shd = shd
                    best_mse_loss = mse
                elif shd < best_shd:
                    best_shd = shd
                    best_shd_graph = graph
                    best_mse_loss = np.inf
                elif shd == best_shd and mse < best_mse_loss:
                    best_mse_loss = mse
                    best_mse_data = preds
                    best_epoch = epoch
                    self.save_model(best_shd_graph, best_mse_data, data)
                else:
                    if best_mse_loss == np.inf:
                        best_mse_loss = mse
                    elif mse < best_mse_loss:
                        best_mse_loss = mse
                        best_mse_data = preds
                
            mse_train.append(mse)
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            
        if ground_truth_G != None:
            
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
                  'time: {:.4f}s'.format(time.time() - t))

            return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A, best_shd, best_shd_graph, best_mse_loss, best_mse_data, best_epoch
        else:
            
            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'time: {:.4f}s'.format(time.time() - t))
            
            return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A
        
    
    def fit(self, train_loader, ground_truth_G = None):
        
        if not hasattr(self, "discriminator"):
            self.discriminator = Discriminator(self.data_variable_size, (256, 256), self.negative_slope).double().to(self.device)
            
        if not hasattr(self, "encoder"): 
            self.encoder = MLPEncoder(self.x_dims, self.encoder_hidden, int(self.z_dims), self.adj_A,
                                  self.device, self.data_type).double().to(self.device)
            
        if not hasattr(self, "decoder"):
            self.decoder = MLPDecoder(self.z_dims, self.x_dims, self.decoder_hidden,
                                  self.device, self.data_type).double().to(self.device)
            
        if not hasattr(self, "optimizerD"):
            self.optimizerD = optim.Adam(
                self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9),
                weight_decay=1e-6)
            
        if not hasattr(self, "optimizerV"):
            self.optimizerV = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
            
        if not hasattr(self, "schedulerV"):
            self.schedulerV = lr_scheduler.StepLR(self.optimizerV, step_size=self.lr_decay, gamma=self.gamma)
            
        if not hasattr(self, "schedulerD"):
            self.schedulerD = lr_scheduler.StepLR(self.optimizerD, step_size=self.lr_decay, gamma=self.gamma)

        best_ELBO_loss = np.inf
        best_NLL_loss = np.inf
        best_MSE_loss = np.inf
        best_shd = np.inf
        best_epoch = 0
        best_ELBO_graph = []
        best_NLL_graph = []
        best_MSE_graph = []
        best_shd_graph = []
        best_MSE_data = []

        try:
            for step_k in range(self.k_max_iter):
                while self.c_A < 1e+20:
                    for epoch in range(self.epochs):
                        (ELBO_loss, NLL_loss, MSE_loss, graph, origin_A, best_shd, best_shd_graph,
                         best_MSE_loss, best_MSE_data, best_epoch) = self.train(train_loader,
                        epoch, best_ELBO_loss, best_epoch, best_MSE_loss, best_MSE_data,
                        best_shd, best_shd_graph, ground_truth_G, 
                        self.lambda_A, self.c_A, self.optimizerV, self.optimizerD)
                        if ELBO_loss < best_ELBO_loss:
                            best_ELBO_loss = ELBO_loss
                            best_epoch = epoch
                            best_ELBO_graph = graph

                        if NLL_loss < best_NLL_loss:
                            best_NLL_loss = NLL_loss
                            best_epoch = epoch
                            best_NLL_graph = graph

                        if MSE_loss < best_MSE_loss:
                            best_MSE_loss = MSE_loss
                            best_epoch = epoch
                            best_MSE_graph = graph

                    print("Optimization Finished!")
                    print("Best Epoch: {:04d}".format(best_epoch))
                    print("Best SHD: {:04d}".format(best_shd))
                    print("Best MSE Loss: {:.10f}".format(best_MSE_loss))
                
                    if ELBO_loss > 2 * best_ELBO_loss:
                        break

                    # update parameters
                    A_new = origin_A.data.clone()
                    self.h_A_new = _h_A(A_new, self.data_variable_size, self.device)
                    if self.h_A_new.item() > 0.25 * self.h_A_old:
                        self.c_A*=10
                    else:
                        break

                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
                self.h_A_old = self.h_A_new.item()
                self.lambda_A += self.c_A * self.h_A_new.item()

                if self.h_A_new.item() <= self.h_tol:
                    break
                
            if ground_truth_G != None:
                # test()
                #print (best_ELBO_graph)
                #print(nx.to_numpy_array(ground_truth_G))
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
                print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                #print(best_NLL_graph)
                #print(nx.to_numpy_array(ground_truth_G))
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
                print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                #print (best_MSE_graph)
                #print(nx.to_numpy_array(ground_truth_G))
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
                print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                #graph = origin_A.data.clone().cpu().numpy()
                # best_shd_graph[np.abs(best_shd_graph) < 0.1] = 0
                # # print(graph)
                # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_shd_graph))
                # print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                # best_shd_graph[np.abs(best_shd_graph) < 0.2] = 0
                # # print(graph)
                # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_shd_graph))
                # print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

                best_shd_graph[np.abs(best_shd_graph) < 0.3] = 0
                # print(graph)
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_shd_graph))
                print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
                
                #graph = origin_A.data.clone().cpu().numpy()
                #graph[np.abs(graph) < self.graph_threshold] = 0
                return best_shd_graph, best_MSE_data
            else:
                graph = origin_A.data.clone().cpu().numpy()
                graph[np.abs(graph) < self.graph_threshold] = 0
                return graph

        except KeyboardInterrupt:
            # print the best anway
            #print(best_ELBO_graph)
            #print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
            print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            #print(best_NLL_graph)
            #print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
            print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            #print(best_MSE_graph)
            #print(nx.to_numpy_array(ground_truth_G))
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
            print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            graph = origin_A.data.clone().cpu().numpy()
            graph[np.abs(graph) < 0.1] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
            print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            graph[np.abs(graph) < 0.2] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
            print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

            graph[np.abs(graph) < 0.3] = 0
            # print(graph)
            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
            print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
    
    def save_model(self, causal_graph, data, real_data):
        assert self.save_directory != '', 'Saving directory not specified! Please specify a saving directory!'
        torch.save(self.encoder.state_dict(), os.path.join(self.save_directory,'encoder.pth'))
        torch.save(self.decoder.state_dict(), os.path.join(self.save_directory,'decoder.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_directory,'discriminator.pth'))
        pd.DataFrame(causal_graph).to_csv(os.path.join(self.save_directory, "adjacency_matrix.csv"), index=False)
        pd.DataFrame(data.clone().squeeze().detach().cpu().numpy()).to_csv(os.path.join(self.save_directory, "generated_data.csv"), index=False)
        pd.DataFrame(real_data.clone().squeeze().detach().cpu().numpy()).to_csv(os.path.join(self.save_directory, "real_data.csv"), index=False)
                
        
    def load_model(self):
        assert self.load_directory != '', 'Loading directory not specified! Please specify a loading directory!'
        
        encoder = MLPEncoder(self.x_dims, self.encoder_hidden, int(self.z_dims), self.adj_A,
                                  self.device, self.data_type).double().to(self.device)
            
        decoder = MLPDecoder(self.z_dims, self.x_dims, self.decoder_hidden,
                                  self.device, self.data_type).double().to(self.device)
            
        discriminator = Discriminator(self.data_variable_size, (256, 256)).double().to(self.device)
            
            
        encoder.load_state_dict(torch.load(os.path.join(self.load_directory,'encoder.pth')))
        decoder.load_state_dict(torch.load(os.path.join(self.load_directory,'decoder.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(self.load_directory,'discriminator.pth')))
            
            
        return encoder, decoder, discriminator