
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
# from tensorflow.keras.utils import to_categorical # Function to convert labels to one-hot encoding
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Activation,Input
# from tensorflow.keras.optimizers import SGD
from PIL import Image
from pathlib import Path
import pickle

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

folder = "resized_gimli/"

labels = np.asarray(["Magni", "Gimli"])
pickle_gimli_name = "gimli_data_as_numpy.pickle"

if not os.path.isfile(pickle_gimli_name):

    print("Creating Gimli Data...")

    list_of_files = [f for f in listdir(folder) if isfile(join(folder, f))]

    data_as_image = [Image.open(str(Path(folder + image).resolve()))
                     for image in list_of_files]

    data_as_np_array = [np.asarray(image) for image in data_as_image]

    gimli_data = np.array(data_as_np_array)

    with open(pickle_gimli_name, 'wb') as f:
        pickle.dump(gimli_data, f)

print("Loading pickle data...")

with open(pickle_gimli_name, 'rb') as f:
    gimli_data = pickle.load(f)

print("Done!")

print("Development set")
print("Images: ", gimli_data.shape)
print("Images type: ", gimli_data.dtype)
print("Labels shape:", labels.shape)
print("\nNumber of classes:", np.unique(labels).size)
print("\nTest set")
print("Images: ", gimli_data.shape)
print("Labels shape: ", labels.shape)
print("Max pixel value: ", np.amax(gimli_data[0]))

# #Number of classes
# num_classes = np.unique(labels).size

# sample_indexes = np.random.choice(np.arange(gimli_data.shape[0], dtype = int),size = 10, replace = False)

# plt.figure(figsize = (24,18))
# for (ii,jj) in enumerate(sample_indexes):
#     plt.subplot(5,6,ii+1)
#     plt.imshow(gimli_data[jj])
#     plt.title("Label: {}".format(labels[1]))
# plt.show()

# train_mean, train_std = gimli_data.mean(),gimli_data.std()

gimli_red = gimli_data[:, :, :, 0]
gimli_green = gimli_data[:, :, :, 1]
gimli_blue = gimli_data[:, :, :, 2]

print("Gimli RED shape: ", gimli_red.shape)
print("Gimli GREEN shape: ", gimli_green.shape)
print("Gimli BLUE shape: ", gimli_blue.shape)

gimli_mean_red = np.mean(gimli_red)
gimli_std_red = np.std(gimli_red)

gimli_mean_green = np.mean(gimli_green)
gimli_std_green = np.std(gimli_green)

gimli_mean_blue = np.mean(gimli_blue)
gimli_std_blue = np.std(gimli_blue)

print("gimli_mean_red value: ", gimli_mean_red)
print("gimli_mean_green value: ", gimli_mean_green)
print("gimli_mean_blue value: ", gimli_mean_blue)

gimli_red_standardized = (gimli_red - gimli_mean_red)/gimli_std_red
gimli_green_standardized = (gimli_green - gimli_mean_green)/gimli_std_green
gimli_blue_standardized = (gimli_blue - gimli_mean_blue)/gimli_std_blue

print("Gimli red mean value = {}".format(np.mean(gimli_red_standardized)))
print("Gimli red std value = {}".format(np.std(gimli_red_standardized)))

print("Gimli green mean value = {}".format(np.mean(gimli_red_standardized)))
print("Gimli green std value = {}".format(np.std(gimli_red_standardized)))

print("Gimli blue mean value = {}".format(np.mean(gimli_red_standardized)))
print("Gimli blue std value = {}".format(np.std(gimli_red_standardized)))

print("Gimli red mean shape = {}".format(gimli_red_standardized.shape))
print("Gimli green mean shape = {}".format(gimli_green_standardized.shape))
print("Gimli blue mean shape = {}".format(gimli_blue_standardized.shape))

standardized_out = np.stack(
    (gimli_red_standardized, gimli_green_standardized, gimli_blue_standardized), axis=3)

print("standardized_out shape = {}".format(standardized_out.shape))
