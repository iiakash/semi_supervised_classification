import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import functions
import networks
import dataset

import sklearn
import os

import matplotlib.pyplot as plt
import seaborn as sns
import time

def train(self, network, network_criterion, lr, weight_decay, tr_loader, tr_dataset_length, Adam = True, scheduler = True):

  since = time.time()
  print('Training the network {}'.format(network.__class__.__name__))
  print('Network Architecture \n{}'.format(network))
  print('Network Criterion {}'.format(network_criterion))
  list_of_network_loss = []
  list_of_clustering_loss = []
  list_of_total_loss = []
  list_of_losses = []
  learning_rates = []
  list_of_centers = []
  list_of_ranks_of_center_distances = []
  list_of_center_distances = []
  list_of_average_acc = []

  if Adam:
      optimizer = torch.optim.Adam(network.parameters(), lr = lr, weight_decay = weight_decay)

  else:
      optimizer = torch.optim.SGD(network.parameters(), lr = lr, momentum = 0.0, weight_decay = weight_decay, nesterov = False)

  for epoch in range(self.n_epochs):

      embedded_representation = []
      batched_center_index = 0                                                
      total_combined_loss = 0.0
      total_network_loss = 0.0
      total_clustering_loss = 0.0
      labels = np.empty((1), int)

      for data, label in tr_loader:

          #extract the sequence and label from the batch and make predictions and return bottleneck

          sequences = data                                    
          batch_labels = label
          labels = np.append(labels, batch_labels.numpy(), axis = 0)
          target_sequences = sequences.clone()
          predictions, bottleneck = self.network(sequences)
          embedded_representation.append(bottleneck.clone().detach())
          batch_embeddings = torch.cat(embedded_representation) 

          #compute the network loss

          network_loss = self.network_criterion(predictions, target_sequences)

          #set condition for pretrain mode

          if epoch <= self.no_of_pretrain_epochs:

              #pretrain mode

              clustering_loss = torch.zeros([1,1], dtype = torch.float64)
              combined_loss = network_loss      # + self.alpha*clustering_loss   # defining the combined loss
              optimizer.zero_grad()

              #calculating the gradients and taking step with only network loss as the clustering loss is zero'

              combined_loss.backward(retain_graph = True)                     # retaining the pytorch computation graph so that backward can be done twice
              optimizer.step()



          else:

              #joint training mode

              clustering_loss = self.clustering_criterion(bottleneck, batched_center_designation[batched_center_index])
              batched_center_index += 1                                       # incrementing the batched center index
              combined_loss = (1- self.alpha)*network_loss + self.alpha*clustering_loss
              optimizer.zero_grad()

              #calculating the gradients but not taking step

              combined_loss.backward(retain_graph = True)

              #updating the weights of the clustering friendly channels wrt combined loss

              bottleneck_layer = functions.get_bottleneck_name(self.network)

              #train_reporter.print_grads(network)

              with torch.no_grad():

                  for name, parameters in self.network.named_parameters():

                      if name == bottleneck_layer:

                          ranked_channels = torch.from_numpy(ranks_of_center_distances)
                          parameters.grad[torch.where(ranked_channels <= self.no_of_clustering_channels)] = 0.0

              optimizer.step()

              #updating the weights of rest of the channels wrt network loss'

              optimizer.zero_grad()
              network_loss.backward()

              with torch.no_grad():

                  for name, parameters in self.network.named_parameters():

                      if name == bottleneck_layer:

                          ranked_channels = torch.from_numpy(ranks_of_center_distances)
                          parameters.grad[torch.where(ranked_channels > self.no_of_clustering_channels)] = 0.0

              optimizer.step()



          total_network_loss += network_loss.item()
          total_clustering_loss += clustering_loss.item()
          total_combined_loss += combined_loss.item()
      #extract embeddings
      embeddings = batch_embeddings

      #make list of losses

      list_of_network_loss.append(total_network_loss/(tr_dataset_length)/self.batch_size)
      list_of_clustering_loss.append(total_clustering_loss/(tr_dataset_length)/self.batch_size)
      list_of_total_loss.append(total_combined_loss/(tr_dataset_length)/self.batch_size)

      #make cluster update interval array

      cluster_update = np.arange(self.no_of_pretrain_epochs, self.n_epochs, self.cluster_update_interval)

      #clustering
      for update in cluster_update:

          if update == epoch:
              since = time.time()
              print('Updating Cluster Centers')
              center_designation_pre = []
              cluster_label_pre = []
              centers_pre = []
              list_of_interim_acc = []
              no_of_channels = embeddings.shape[1]

              for i in range(no_of_channels):
                  channel = embeddings[:,i,:].numpy()
                  choice_cluster, initial_centers, cluster_ass = functions.kmeansalter(channel, self.n_clusters)
                  labels[np.where(labels > 0)] = 1
                  interim_acc = functions.metrics.acc(labels[1:].flatten(), choice_cluster)
                  list_of_interim_acc.append(interim_acc)
                  cluster_label_pre.append(torch.from_numpy(choice_cluster).unsqueeze(0).transpose(1,0))
                  cluster_label = torch.cat(cluster_label_pre, dim = 1)
                  centers_pre.append(torch.from_numpy(initial_centers).unsqueeze(0).transpose(1,0))
                  centers = torch.cat(centers_pre, dim = 1)
                  center_designation_pre.append(cluster_ass.unsqueeze(0).transpose(1,0))
                  center_designation = torch.cat(center_designation_pre, dim = 1)

              average_acc = (np.sum(functions.averaged_acc(np.asarray(list_of_interim_acc), self.no_of_clustering_channels))*100)/self.no_of_clustering_channels
              list_of_average_acc.append(average_acc)
              batched_center_designation = list(functions.divide_batches(center_designation, self.batch_size))
              center_distances, ranks_of_center_distances = functions.rank_channels(centers)
              end = time.time()
              hours, minutes, seconds = functions.timer(since, end)
              print("Time taken {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

      print('Epoch : {}/{} Network Loss : {} Clustering Loss : {} Total Loss : {}'.format(epoch+1, 
        self.n_epochs, (total_network_loss/(tr_dataset_length/self.batch_size)), 
    (total_clustering_loss/(tr_dataset_length/self.batch_size)),
    (total_combined_loss/(tr_dataset_length/self.batch_size))))

  list_of_centers.append(centers.numpy())
  list_of_ranks_of_center_distances.append(ranks_of_center_distances)
  list_of_center_distances.append(center_distances)
  list_of_losses.append(list_of_network_loss)
  list_of_losses.append(list_of_clustering_loss)
  list_of_losses.append(list_of_total_loss)

  return self.network, optimizer, list_of_network_loss, list_of_clustering_loss,
list_of_total_loss, list_of_losses, embeddings, labels, list_of_centers, list_of_ranks_of_center_distances,
list_of_center_distances, list_of_average_acc
