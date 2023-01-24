# Zero-Shoot Image Clasification

This repository contains the code for the zero-shot image classification method. This work is strongly based on the paper [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600). 

The selected model correspond to [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) available in huggingface.

## About Zero-Shot image classification

Zero-shot image classification is a cutting-edge computer vision task in which a model is trained to recognize new classes of objects without having seen any examples of those classes during training. It is a variation of transfer learning, where the knowledge acquired during the training of one model is transferred to classify novel classes that were not present in the original training data. For instance, a model that has been trained to differentiate cats from dogs can be used to classify images others animal like horses [1](https://huggingface.co/tasks/zero-shot-image-classification).

![Zero-Shot Learning Process. The embedding network trained on a large data set is used to extract a feature vector with which the probability of fitting one of the given labels is calculated. image obtained from [2](https://www.v7labs.com/blog/zero-shot-learning-guide)](zero-shoot/zero-shot classification scheme.png)

In this learning paradigm, the data consists of three essential components: 
- Seen data: Are images and their corresponding labels, which the model uses to learn during training.
- Unseen data: Only consist in labels, and no images are provided.
- Auxiliary information: This can be in the form of textual descriptions or word embeddings and helps the model to generalize to new classes.

The use of zero-shot image classification is crucial as it allows models to learn more efficiently and effectively. Traditional image classification models require large amounts of labeled data for each class to be recognized, which can be time-consuming and expensive to collect. With zero-shot image classification, models can learn to recognize new classes using only a small amount of additional information, such as class attributes or semantic relationships between classes, which is often easier and cheaper to obtain [3](https://arxiv.org/abs/1707.00600).

## About the model

The model correspond to a CLIP (Contrastive Language-Image Pretraining) version, which is a transformer-based model developed by OpenAI in 2021. It is designed to make zero-shot image classification more effective.

The CLIP model consists of two parts: a text transformer for encoding text embeddings and a vision transformer (ViT) for encoding image embeddings. Both models are optimized during pretraining to align similar text and images in vector space. This is achieved by taking image-text pairs and pushing their output vectors closer in vector space while separating the vectors of non-pairs [4](https://www.pinecone.io/learn/zero-shot-image-classification-clip/).

![CLIP Architecture. image obtained from [4](https://www.pinecone.io/learn/zero-shot-image-classification-clip/)](zero-shoot/zero-shot-image-classification-clip.png)

One of the key features of CLIP is its use of a large dataset of 400M text-image pairs scraped from across the internet during its pretraining process. This large dataset size allows CLIP to build a strong understanding of general textual concepts displayed within images. Additionally, CLIP requires only image-text pairs rather than specific class labels, thanks to its contrastive rather than classification-focused training function. This type of data is abundant in today’s social-media-centric world.

In specific, this model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss [5](https://huggingface.co/openai/clip-vit-base-patch32).

### Limitations and biases

A study made by Open AI on the CLIP has found several limitations. One limitation is that the model struggles with tasks such as fine-grained classification and counting objects. Additionally, the study found that the model poses issues with regards to fairness and bias. The study also found that the performance and biases of CLIP depend significantly on the design of classes and the choices made for categories to include and exclude. Furthermore, the study found that disparities in performance could shift based on how the classes were constructed.

In terms of bias and fairness, the study evaluated the model using the Fairface dataset and found significant disparities with respect to race and gender. The study also found that accuracy for gender classification across all races was >96%, with 'Middle Eastern' having the highest accuracy (98.4%) and 'White' having the lowest (96.5%). Additionally, the model averaged ~93% for racial classification and ~63% for age classification. It's worth noting that the aim of the study was to evaluate the performance of the model across people and surface potential risks, not to demonstrate an endorsement/enthusiasm for such tasks. 

For this reason, it is important to be aware of the limitations and biases of the model because they can affect the results of the zero-shot classification.

## About the chosen labels

The labels where manually selected trying to find the official english name of the traffic sign. Nevertheless, not all the labels have an official english name. For that reason, we tried to find the best name for each label. 

The following table shows the selected labels.

| Label | Name | Label | Name | Label | Name |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Speed limit 20 | 1 | Speed limit 30 | 2 | Speed limit 50 |
| 3 | Speed limit 60 | 4 | Speed limit 70 | 5 | Speed limit 80 |
| 6 | End of speed limit 80 | 7 | Speed limit 100 | 8 | Speed limit 120 |
| 9 | No overtaking | 10 | No overtaking for trucks | 11 | Give way |
| 12 | Priority road | 13 | Yield | 14 | Stop |
| 15 | Road closed | 16 | Heavy vehicles prohibited | 17 | Do not enter |
| 18 | General warning | 19 | Left bend | 20 | Right bend |
| 21 | Double bend | 22 | Uneven road | 23 | Slippery road |
| 24 | Narrow road | 25 | Roadworks ahead | 26 | Traffic signals ahead |
| 27 | Pedestrian crossing | 28 | Watch for children | 29 | Bicycle crossing |
| 30 | Snow | 31 | Cattle on the road | 32 | End of all restrictions |
| 33 | Turn right ahead | 34 | Turn left ahead | 35 | Straight ahead only |
| 36 | Ahead or turn right only | 37 | Ahead or turn left only | 38 | Bypass on right |
| 39 | Bypass on left | 40 | Roundabout | 41 | End of no overtaking zone |
| 42 | End of no overtaking zone for trucks |

![Random images from the dataset](zero-shoot/labels.png)

The hypotesis template is the following: "This is an image of a {} road sign."

## Installation

You can find a setup script in the root directory of the repository. The script will create a virtual environment and install all the required packages. The script is tested on Ubuntu 20.04.2.

```bash
./setup.sh
```

## Usage

With the file 'pipeline.py' you can run the zero-shot learning pipeline. The pipeline will download the model and process the imagenes from the given folders. Below you can find the usage for 100 images and the labels number 1 and 2.

```bash

python pipeline.py --labels_num 1 2 --n_images 100

```

The output will be a csv with the three best predictions for each image. The csv will be saved in the 'results' folder.

## Show results

To show some graphs you can use "create_graphs.py". The script will create a bar graph with the accuracy from each label and will save the summary of the results in a csv file.

```bash

python create_graphs.py

```


