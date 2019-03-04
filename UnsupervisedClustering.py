"""
BANERJEE, Rohini (20543577)
MSBD5002: Data Mining and Knowledge Discovery
Assignment 3: Unsupervised Image Clustering
"""

# Importing all the required libraries
import numpy as np
import os, sys,cv2
from keras.applications.nasnet import NASNetMobile
from keras.preprocessing import image
from PIL import Image as pimg
import keras.applications
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create lists to store the images and its features
files=[]
train=[]

# Function to get the image size in the image directory
def file_size(file_name):
    stats=os.stat("images/"+str(file_name))
    return stats.st_size

# Function to call the keras pre-trained model to get feature weights of the images
def nasnet_model():
    base_model = NASNetMobile(weights='imagenet', include_top=True)
    # PLEASE make changes to the source code of nasnet before running the next line
    # Access the fully connected model by adding the same variable (in this case 'rohini') in the source code
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('rohini').output)
    images_path = "./images"
    # loading all the image names into the files list
    for f in os.listdir(images_path):
        files.append(f)
    # Looping over every image present in the files list
    for img_path in files:
        if(file_size(img_path)!=0):
            # load the image and resize it
            img = image.load_img("./images/"+str(img_path), target_size=(224, 224))
            # extract features from each image
            x_image = image.img_to_array(img)
            x_image = np.expand_dims(x_image, axis=0) # increase dimensions of x to make it suitable for further feature extraction
            x_image = keras.applications.nasnet.preprocess_input(x_image)
            x_features = model.predict(x_image) # extract image features from model
            x_features = np.array(x_features) # convert features list to numpy array
            x_flatten= x_features.flatten() # flatten out the features in x
            train.append(x_flatten) # this list contains the final features of the images

# Function to perform PCA on the extracted features of the images
def create_PCA(data_features, n_components=None):
    pca = PCA(n_components=n_components, random_state=728)
    pca_features = pca.fit_transform(data_features)
    return pca_features

# Function to perform K-Means clustering on the feature dataset
def kmeans_clustering(data):
    # the parameter values are chosen after comparing the results of various trails
    # 20 clusters have been chosen to be the optimum value for the given dataset
    kmeans = KMeans(n_clusters=20, init='random', max_iter=500, n_init=5,tol=1e-3, verbose=1,\
                       n_jobs=-1, algorithm='auto', precompute_distances=True, random_state=42)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    labels = kmeans.labels_
    file_label_zip = dict(zip(files,labels)) # zipping the filename and cluster label together
    # writing the images into different clusters
    for i,j in file_label_zip.items() :
        new_path=str("./images/"+str(j))
        # create a new folder if the cluster folder does not exist
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        # read the cluster and image and write it into the appropriate folder
        img = cv2.imread("./images/"+str(i), 1)
        cv2.imwrite(os.path.join(new_path , i), img)


""" -------------- MAIN PROGRAM ----------------------- """
# Call the nasnet model to extract features of the given images
nasnet_model()
# Perform PCA on the extracted features
train_pca = create_PCA(train)
# Perform KMeans clustering on the pre-processed features
kmeans_clustering(train_pca)
""" The following part of the code is to write the clustered images into a csv file in the given format """
main_path = "./images/"
# Count the number of clusters made by the model
folders = 0
for _, dirnames, filenames in os.walk(main_path):
    folders += len(dirnames)
header_str = "Cluster "
header = [] # create list to hold all the header values
cluster_image =[] # save all image names pertaining to a cluster into a list of lists
# Append the header and image names for each cluster created
for i in range(folders):
    header.append(header_str+str(i+1))
    path = "./images/"+str(i)+"/"
    cluster_image.append([str(os.path.splitext(filename)[0]) for filename in os.listdir(path)])
# Convert the names of images into strings for the required CSV format structure
final_cluster_image = []
for cluster in cluster_image:
    str_image = []
    for image in cluster:
        image = "'" + str(image) + "'" # making the image names into strings
        str_image.append(image) # list containing the cluster image names as strings
    final_cluster_image.append(str_image)
# Make all the rows the same size so that it can be transposed easily
max_row_length = max([len(row) for row in final_cluster_image])
output = [row + ['']*(max_row_length - len(row)) for row in final_cluster_image]
output = np.array(output).T # take transpose of matrix for saving the CSV in column wise format
header = ','.join(header) # join to one string for numpy savetxt format
np.savetxt("prediction.csv", output, delimiter=',', header=header,fmt='%s',comments='') # get the required CSV
