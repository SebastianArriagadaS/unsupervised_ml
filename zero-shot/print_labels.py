import os
import matplotlib.pyplot as plt
import random 
# Load the first image of every class and plot it with the label

# get current directory
path = os.getcwd()
parent_path = os.path.dirname(path)

# data path
data_path = f"{parent_path}/data"

# load the labels
with open(f"{path}/labels.txt", "r") as f:
    labels = f.read().splitlines()

# create the plot with the 43 classes
fig, axs = plt.subplots(9, 5, figsize=(30, 30))
fig.subplots_adjust(hspace=0.4, wspace=0.15)
axs = axs.ravel()

# plot the images
for i, label in enumerate(labels):
    # get the path to the images
    images = os.listdir(f"{data_path}/train/{i}")
    # get a random image
    image = random.choice(images)
    image_path = f"{data_path}/train/{i}/{image}"
    # load the image
    image = plt.imread(image_path)
    # plot the image
    axs[i].imshow(image)
    axs[i].set_title(label)
    axs[i].axis("off")

# delete the last 2 plots
for i in range(43, 45):
    fig.delaxes(axs[i])

plt.show()
