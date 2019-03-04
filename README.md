
# Implementation of Unsupervised Image Clustering (Python 3) #
Unsupervised Learning can be defined as a class of Machine Learning where different techniques are used to find patterns in the data. The data feeded to unsupervised algorithms are not labelled. This assignment focuses on unsupervised image clustering, i.e, clustering similar images together. 


## 1. Getting Started ##
Before running this code, please make sure that you download all the images in the link (https://www.dropbox.com/sh/gr2istwq2qrnjy8/AAD2dP8T57hQnDvh1UW-3wZUa?dl=0) and store them in a folder called **images**. This folder needs to be kept in the same directory as the code.
In this assignment, the code reads 5011 images, preprocesses them, extracts features, and then clusters them accordingly. This takes some time, so please be patient. On a GPU enabled system, it will take about 20-25 mins to run.

### 1.1. Prerequisites ###
The following libraries are required for running this code:

1. **numpy**
2. **os**
3. **sys**
4. **cv2**
5. **keras**
6. **sklearn**
7. **PIL**


## 2. Implementation ##
In this implementation of unsupervised image clustering, I have used the Keras NASNet (Neural Architecture Search Network) model, with weights pre-trained on ImageNet. The default input size for the used NASNetMobile model is 224x224.

**NOTE:** This implementation uses NASNet after fully connecting it to the outer layer. Hence few changes might be required in the source code, depending on the device configuration. For my machine, I had to make the change in the source code. To make the change, open the **nasnet.py** file (if you use *anaconda* it will be downloaded there under the *keras* folder) and add *name='XXX'* as a parameter to the *GlobalAveragePooling2D* in line 242.
Now add the same *name* in line 33 of this implementation code (within the **nasnet_model()** function). If the change is not required for your device, please change the **nasnet_model()** function accordingly (i.e., *base_model* will become equivalent to the final *model*).

### 2.1. The NASNet Model Function ###
The **nasnet_model()** function uses the NASNetMobile version of the pretrained keras application. As mentioned before, this model is used for the fully connected outer layer and changes in the source code may be required. The function then reads all the images present in the *images* folder and preprocesses them to extract and flatten its features. The *train* contains all the features of each image as a list of lists.

### 2.2. The Create PCA Function ###
The **create_PCA()** function takes the features extracted by NASNet and performs Principal Component Analysis on them. It then returns the transformed *train_pca* list.

### 2.3. The KMeans Clustering Function ###
The **kmeans_clustering()** function makes use of KMeans to cluster the images based on feature similarity. The function clusters and then stores the images into *k* folders within the *images* folder. After a lot of trial and error, the following hyperparameters were chosen for this assignment:

1. **n_clusters = 20:** Starting from 10-25, the cluster size of 20 gave the most promising result and gave the least size of the worst cluster.
2. **init = 'random':** This parameter gave better results than *'k-means++'* initialization.
3. **max_iter = 500, n_init = 5, tol = 1e-3:** A combination of a relatively low tolerance (*tol*), less running on different centroid seeds (*n_init*) and a moderately high iterations for a single run (*max_iter*) gave the best clusters.
4. **algorithm = 'auto':** This parameter chose the optimum kmeans algorithm for the given dataset. It automatically used *elkan* for dense data and *full* for sparse data.
5. **precompute_distances = True:** Though it takes more memory, this parameter, boosted the cluser structure a lot.

### 2.4. CSV Output ###
Once the clusters are identified, the images are read from the clustered folders within the *images* folder and written onto the CSV file in the prescribed column-wise format. Though the writing of the images into clustered folders is not required, I have implemented that as it is easy to visualize the effectiveness of the clusters.

## 3. Advantages Over Other Models ##
I have chosen a combination of NASNetMobile and KMeans over the other image weight and clustering combinations. This is because the number of images in the worst classified cluster was minimum here. Spectral Clustering, though faster, provided a 'bad' cluster of almost 2500 images. VGG16, RESNet and InceptionNet were not able to cluster distinct items separately (i.e., bus and car were put together, bottles and food were put together, etc.).

## 4. Cluster Inference from the Provided Model ##
The model provided in the assignment, clusters the images based on the following main clusters:

1. **Horses**
2. **Trains**
3. **Bottles**
4. **Buses**
5. **Cars**
6. **Potted Plants**
7. **Sheep and Cows**
8. **Birds and Others:** Worst classified cluster
9. **Dogs**
10. **Sofas and Couches**
11. **Boats and Ships**
12. **Motorbikes**
13. **Food**
14. **Bicycles**
15. **Aeroplanes**
16. **People 1:** Mostly kids
17. **Screens**
18. **Cats**
19. **People 2:** Mostly people indoors
20. **People 3:** Mostly people outdoors

## 5. Authors ##
BANERJEE, Rohini - HKUST Student ID: 20543577
