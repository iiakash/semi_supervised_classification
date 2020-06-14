import time
import numpy as np
        
import dcca
import networks_ss
import torch
import torch.nn as nn
import helpers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

run = dcca.DCCA(mode, network, network_criterion, n_clusters, 
       clustering_criterion, cluster_update_interval,
       no_of_clustering_channels, n_epochs, no_of_pretrain_epochs, 
       batch_size, lr, alpha,downsampling_step, sequence_length, kernel_size)

train_loader, test_loader, _ = run.load_teastman_data(normalize = True)


network, optimizer, list_of_network_loss, list_of_clustering_loss,list_of_total_loss,
list_of_losses, embeddings, labels, list_of_centers, list_of_ranks_of_center_distances,
list_of_center_distances, list_of_average_acc = run.train_teastman(train_loader, 9920, Adam = False, scheduler = True)

labels_pred = run.training_predictions(embeddings)

labels[np.where(labels > 0)] = 1  
list_of_nmi, list_of_ari, list_of_acc, list_of_cm = run.calculate_metrics(labels[1:], labels_pred)

classifier = networks_ss.Classifier(network)
classifier_criterion = nn.CrossEntropyLoss()
classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr = 0.001, momentum = 0.9)
classifier_epoch = 200
list_of_classification_loss_epoch = []
list_of_classification_accuracy_epoch = []
list_of_precision_epoch = []
list_of_recall_epoch = []
list_of_classification_loss_batch = []
list_of_classification_accuracy_batch = []
list_of_precision_batch = []
list_of_recall_batch = []

for epoch in range(classifier_epoch):

    running_loss = 0.0
    running_classification_accuracy = 0.0
    running_precision = 0.0
    running_recall = 0.0
    batch_counter = 0
    for data, label in train_loader:
        batch_counter += 1
        output = classifier(data)
        classification_loss = classifier_criterion(output, label)
        classification_loss.backward()
        classifier_optimizer.step()
        classifier_optimizer.zero_grad()
        running_loss += classification_loss.item()
        _, prediction = torch.max(output.clone().detach(), 1)
        batch_classification_accuracy = accuracy_score(prediction, label)
        batch_precision = precision_score(prediction, label, average = 'macro')
        batch_recall = recall_score(prediction, label, average = 'macro')
        running_classification_accuracy += batch_classification_accuracy
        running_precision += batch_precision
        running_recall += batch_recall
        list_of_classification_loss_batch.append(running_loss)
        list_of_classification_accuracy_batch.append(running_classification_accuracy)
        list_of_precision_batch.append(running_precision)
        list_of_recall_batch.append(running_recall)
    epoch_loss = running_loss / batch_counter
    epoch_classification_accuracy = running_classification_accuracy / batch_counter
    epoch_precision = running_precision / batch_counter
    epoch_recall = running_recall / batch_counter
    list_of_classification_loss_epoch.append(epoch_loss)
    list_of_classification_accuracy_epoch.append(epoch_classification_accuracy)
    list_of_precision_epoch.append(epoch_precision)
    list_of_recall_epoch.append(epoch_recall)
        
    print('Epoch {}/{} Classification Loss {} Accuracy {} Precision {}
    Recall {}'.format(epoch+1, classifier_epoch, epoch_loss, epoch_classification_accuracy, epoch_precision, epoch_recall))
