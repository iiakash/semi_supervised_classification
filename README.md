# Semi Supervised Classifier using Pretrained Autoencoder

## The reopository contains a semi supervised approach of classifying faults in the Tennessee Eastman Process and the implementation with PyTorch

## Overview

One of the major challenges when it comes to dealing with real world data is the amount of noise and unnecessary features that are present in the data. If these noise and unnecessary samples are not handled and the data is not cleaned properly before feeding them into the machine learning model, it can result in poor performance of the trained model. To address this issue in this project a semi supervised method of classification is presented, where the training is divided into two parts. The first part consists of unsupervised training of the model, where most of the noise is dealt with. The unsupervised network only learns the salient features in the data while ignoring the less important features and samples. This trained model is then used as a classifier by adding multiple layers and training again in a supervised manner.

## Data

The data used for this project is from the Tennessee Eastman Process. The Tennessee Eastman Process is a simulation of a chemical plant which is based on an actual process in the Eastman Chemical Company located in Tennessee, USA. The Tennessee Eastman process was originally created by Downs and Vogel as a process control challenge problem. 

The generated dataset from the Tennessee Eastman Process consists of 22 continuous process measurements, 19 component analysis measurements, and 12 manipulated variables. The dataset consists of 21 pre-programmed faults, among which 16 are known fault cases, and 5 fault cases are unknown. Both the training and testing datasets include a total of 52 observed variables. The training dataset consists of 22 different simulation runs, and simulation 0 is fault free. In our case, this simulation is considered as our normal data samples. Simulations 1 to 21 were generated for 21 fault cases, and in our case all of these 21 simulations are considered as anomalous data samples. Similarly, the testing data set contains 22 different simulations, the first one being the normal case, and the rest are simulations for different fault cases. All of the 22 data sets have 960 observations each, and 52 observed variables.


## Dependencies

All the dependencies are listed in the requirements.txt file.
