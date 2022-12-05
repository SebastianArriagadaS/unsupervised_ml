# ADND-J7-unsupervised-ml

This repository contains the code for the unsupervised machine learning project of the Advanced Deep Neural Networks course at the University of Eotvos Lorand, Budapest.

This project is based on the German traffic sign recognition benchmark dataset. The dataset contains 43 different classes of traffic signs. The goal of the project is find the best unsupervised learning method to cluster the traffic signs into the 43 classes.

## Installation

You can find a setup script in the root directory of the repository. The script will create a virtual environment and install all the required packages. The script is tested on Ubuntu 20.04.2.

```bash
./setup.sh
```

## Usage

With the file 'pipeline.py' you can run the zero-shot learning pipeline. The pipeline will download the model and process the imagenes from the given folders. Below you can find the usage for 100 images and the labels number 1 and 2.

```bash

python pipeline.py --labels_num 1 2 --num_images 100

```

The output will be a csv with the three best predictions for each image. The csv will be saved in the 'results' folder.

## Show results

To show some graphs you can use "create_graphs.py". The script will create a bar graph with the accuracy from each label and will save the summary of the results in a csv file.

```bash

python create_graphs.py

```

Until the moment you can see the summary results in the following table:

| Label | Accuracy 1st | Accuracy 2nd acc | Accuracy 3rd acc | 
| :---: | :---: | :---: | :---: |
| Maximum speed 20 | 86 | 97 | 99 |
| Stop | 93 | 94 | 95 |
| Heavy Vehicles prohibited | 76 | 89 | 93 |
| Maximum speed 70 | 91 | 93 | 94 |
| Maximum speed 60 | 36 | 75 | 85 |
| Maximum speed 30 | 48 | 68 | 83 |
| Watch for children | 59 | 71 | 80 |
| Maximum speed 50 | 71 | 76 | 77 |
| End maximum speed 80 | 44 | 62 | 76 |
| Maximum speed 100 | 61 | 66 | 70 |
| Yield | 43 | 60 | 67 |
| Maximum speed 80 | 4 | 28 | 48 |
| Pedestrian crossing | 11 | 27 | 37 |
| End of no passing zone | 4 | 10 | 16 |
| Turn right ahead | 5 | 14 | 24 |
| End of all restrictions | 2 | 13 | 18 |
| Ahead or turn left only | 0 | 3 | 10 |
| No passing for trucks | 0 | 5 | 6 |
| Cattle | 2 | 3 | 4 |
| Ahead or turn right only | 0 | 1 | 3 |
| No passing | 2 | 3 | 8 |
| Ahead only | 0 | 1 | 1 |

The performance of this method depends a lot from the selected label names. For that reason, we tried to find the best label names for each label. The following table the different tries for the labels with worst performance.

TODO

