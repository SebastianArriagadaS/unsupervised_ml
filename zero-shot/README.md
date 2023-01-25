# Zero-shot Image Clasification

This repository contains the code for the zero-shot image classification method. This work is strongly based on the paper [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600). 

The selected model correspond to [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) available in huggingface.

## About Zero-Shot image classification

Zero-shot image classification is a cutting-edge computer vision task in which a model is trained to recognize new classes of objects without having seen any examples of those classes during training. It is a variation of transfer learning, where the knowledge acquired during the training of one model is transferred to classify novel classes that were not present in the original training data. For instance, a model that has been trained to differentiate cats from dogs can be used to classify images others animal like horses [[1]](https://huggingface.co/tasks/zero-shot-image-classification).

| ![zero-shot%20classification%20scheme.png](https://github.com/SebastianArriagadaS/unsupervised_ml/blob/main/zero-shot/zero-shot%20classification%20scheme.png) | 
|:--:| 
| *The process of zero-shot learning involves the use of an embedding network trained on a large dataset to extract a feature vector. This feature vector is then used to calculate the probability of the input fitting one of the given labels. image obtained from [www.v7labs.com](https://www.v7labs.com/blog/zero-shot-learning-guide)* |

In this learning paradigm, the data consists of three essential components: 
- **Seen data** Are images and their corresponding labels, which the model uses to learn during training.
- **Unseen data** consist only in labels, and no images are provided.
- **Auxiliary information** can be in the form of textual descriptions or word embeddings and helps the model to generalize to new classes.

The formal definition of zero-shot learning is that given labeled training instances D<sub>tr</sub> belonging to the seen classes S, the aim is to learn a classifier f<sup>u</sup> (⋅) : X → U that can classify testing instances X<sub>te</sub>  (i.e., to predict Y<sub>te</sub>) belonging to the unseen classes U. The classifier f<sup>u</sup> (⋅) is trained using the auxiliary information A, which is a set of attributes or textual descriptions of the seen classes [[2]](https://arxiv.org/abs/1707.00600).

The use of zero-shot image classification is crucial as it allows models to learn more efficiently and effectively. Traditional image classification models require large amounts of labeled data for each class to be recognized, which can be time-consuming and expensive to collect. With zero-shot image classification, models can learn to recognize new classes using only a small amount of additional information, such as class attributes or semantic relationships between classes, which is often easier and cheaper to obtain [[2]](https://arxiv.org/abs/1707.00600).

## About the model

The model correspond to a CLIP (Contrastive Language-Image Pretraining) version, which is a transformer-based model developed by OpenAI in 2021. It is designed to make zero-shot image classification more effective.

The CLIP model consists of two parts: a text transformer for encoding text embeddings and a vision transformer (ViT) for encoding image embeddings. Both models are optimized during pretraining to align similar text and images in vector space. This is achieved by taking image-text pairs and pushing their output vectors closer in vector space while separating the vectors of non-pairs [[3]](https://www.pinecone.io/learn/zero-shot-image-classification-clip/).

| ![zero-shot-image-classification-clip.png](https://github.com/SebastianArriagadaS/unsupervised_ml/blob/main/zero-shot/zero-shot-image-classification-clip.png) | 
|:--:| 
| *CLIP model architecture summary. image obtained from [https://www.pinecone.io](https://www.pinecone.io/learn/zero-shot-image-classification-clip/)* |

One of the key features of CLIP is its use of a large dataset of 400M text-image pairs scraped from across the internet during its pretraining process. This large dataset size allows CLIP to build a strong understanding of general textual concepts displayed within images. Additionally, CLIP requires only image-text pairs rather than specific class labels, thanks to its contrastive rather than classification-focused training function. This type of data is abundant in today’s social-media-centric world.

In specific, this model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss [[4]](https://huggingface.co/openai/clip-vit-base-patch32).

### Limitations and biases

A study made by Open AI on the CLIP has found several limitations. One limitation is that the model struggles with tasks such as fine-grained classification and counting objects. Additionally, the study found that the model poses issues with regards to fairness and bias. The study also found that the performance and biases of CLIP depend significantly on the design of classes and the choices made for categories to include and exclude. Furthermore, the study found that disparities in performance could shift based on how the classes were constructed.

In terms of bias and fairness, the study evaluated the model using the Fairface dataset and found significant disparities with respect to race and gender. The study also found that accuracy for gender classification across all races was >96%, with 'Middle Eastern' having the highest accuracy (98.4%) and 'White' having the lowest (96.5%). Additionally, the model averaged ~93% for racial classification and ~63% for age classification. It's worth noting that the aim of the study was to evaluate the performance of the model across people and surface potential risks, not to demonstrate an endorsement/enthusiasm for such tasks [[1]](https://huggingface.co/tasks/zero-shot-image-classification). 

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

| ![zero-shot/labels.png](https://github.com/SebastianArriagadaS/unsupervised_ml/blob/main/zero-shot/labels.png) | 
|:--:| 
| *Random selected images from the dataset.* |

The auxiliary information hypothesis template is as follows: "This is an image of a {} road sign."

## Performance estimation

The performance of the model was estimated on base of the confusion matrix. A confusion matrix is a table that is often used to describe the performance of a classification algorithm, such as a machine learning model. It is typically used to describe the performance of a model on a set of test data for which the true values are known.

The columns of the matrix represent the predicted class, while the rows represent the true class. The entries in the matrix represent the number of observations that were predicted to belong to a certain class, but were actually of a different class.

| ![zero-shot/confusion_matrix_scheme.png](https://github.com/SebastianArriagadaS/unsupervised_ml/blob/main/zero-shot/confusion_matrix_scheme.png) | 
|:--:| 
| * Confusion matrix scheme. image obtained from [https://en.wikipedia.org](https://en.wikipedia.org/wiki/Confusion_matrix)* |

- **Accuracy** is the ratio of correctly predicted observation to the total observations. ( (True Positive + True Negative) / Total )

- **Precision** is the ratio of correctly predicted positive observations to the total predicted positive observations. (True Positive / (True Positive + False Positive))

- **Recall (Sensitivity)** is the ratio of correctly predicted positive observations to the all observations in actual class. (True Positive / (True Positive + False Negative))

- **F1 Score** is the Harmonic Mean between precision and recall. The range for F1 Score is [0, 1]. It tells you how precise your classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances). The greater the F1 Score, the better is the performance of our model.

All of these metrics are important for evaluating the performance of a classification model, and can help to understand where the model is making errors and how to improve it.

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

To show some graphs you can use "create_confusion_matrix.py". The script will create a bar graph with the accuracy from each label and will save the summary of the results in a csv file.

```bash

python create_confusion_matrix.py

```

| Label | Accuracy | Precision | Recall | F1_score |
|-------|----------|-----------|--------|----------|
| **Global resutls** |  **24.1** | **28.5** |  **24.1** |  **22.2** |
| Speed limit 20 | 78.7 | 82.3 | 93.0 | 87.3 |
| Speed limit 30 | 60.2 | 63.3 | 81.0 | 71.1 |
| Speed limit 50 | 54.6 | 84.0 | 21.0 | 33.6 |
| Speed limit 60 | 46.9 | 45.0 | 59.0 | 51.1 |
| Speed limit 70 | 76.9 | 86.5 | 83.0 | 84.7 |
| Speed limit 80 | 55.9 | 69.1 | 38.0 | 49.0 |
| End of speed limit 80 | 44.1 | 32.9 | 26.0 | 29.1 |
| Speed limit 100 | 69.4 | 86.8 | 66.0 | 75.0 |
| Speed limit 120 | 59.5 | 71.6 | 53.0 | 60.9 |
| No overtaking | 49.5 | 33.3 | 2.0 | 3.8 |
| No overtaking for trucks | 33.4 | 2.9 | 3.0 | 2.9 |
| Give way | 47.4 | 0.0 | 0.0 | 0.0 |
| Priority road | 49.0 | 25.0 | 2.0 | 3.7 |
| Yield | 50.5 | 75.0 | 3.0 | 5.8 |
| Stop | 86.2 | 89.6 | 95.0 | 92.2 |
| Road closed | 50.0 | 0.0 | 0.0 | 0.0 |
| Heavy vehicles prohibited | 17.8 | 15.4 | 81.0 | 25.9 |
| Do not enter | 48.3 | 29.4 | 5.0 | 8.5 |
| General warning | 9.9 | 8.1 | 78.0 | 14.6 |
| Left bend | 46.3 | 0.0 | 0.0 | 0.0 |
| Right bend | 46.5 | 0.0 | 0.0 | 0.0 |
| Double bend | 46.7 | 0.0 | 0.0 | 0.0 |
| Uneven road | 33.9 | 3.9 | 4.0 | 3.9 |
| Slippery road | 27.5 | 15.6 | 37.0 | 22.0 |
| Narrow road | 44.8 | 4.0 | 1.0 | 1.6 |
| Roadworks ahead | 36.8 | 17.3 | 19.0 | 18.1 |
| Traffic signals ahead | 48.3 | 0.0 | 0.0 | 0.0 |
| Pedestrian crossing | 41.8 | 35.3 | 47.0 | 40.3 |
| Watch for children | 16.5 | 1.7 | 7.0 | 2.7 |
| Bicycle crossing | 63.3 | 95.7 | 44.0 | 60.3 |
| Snow | 57.5 | 86.1 | 31.0 | 45.6 |
| Cattle on the road | 46.1 | 9.5 | 2.0 | 3.3 |
| End of all restrictions | 48.8 | 0.0 | 0.0 | 0.0 |
| Turn right ahead | 41.5 | 20.3 | 14.0 | 16.6 |
| Turn left ahead | 32.7 | 10.4 | 14.0 | 12.0 |
| Straight ahead only | 49.8 | 0.0 | 0.0 | 0.0 |
| Ahead or turn right only | 44.6 | 0.0 | 0.0 | 0.0 |
| Ahead or turn left only | 42.6 | 0.0 | 0.0 | 0.0 |
| Bypass on right | 39.4 | 0.0 | 0.0 | 0.0 |
| Bypass on left | 40.3 | 10.0 | 6.0 | 7.5 |
| Roundabout | 35.7 | 16.7 | 20.0 | 18.2 |
| End of no overtaking zone | 37.7 | 1.5 | 1.0 | 1.2 |
| End of no overtaking zone for trucks | 44.6 | 0.0 | 0.0 | 0.0 |

If we focus on the [confusion matrix](https://github.com/SebastianArriagadaS/unsupervised_ml/tree/main/zero-shot/confusion_matrix.png), we can observe that most errors occur in False Positives with the label "General Warning", this can be due to the model having learned the general concept of a traffic sign. Along with this, it is important to consider that there are two types of patterns that repeat in all signs, which are the circular and triangular shapes. This can make recognition more difficult, as to differentiate them, the model has to focus mainly on the center area.

| ![confusion_matrix.png](https://github.com/SebastianArriagadaS/unsupervised_ml/blob/main/zero-shot/confusion_matrix.png) | 
|:--:| 
| *Confusion matrix* |

Therefore, this method may be more useful for research on general features, for example, the results presented below show that if we only differentiate between triangular and circular signs, we obtain very good results.

| Label | Accuracy | Precision | Recall | F1_score |
|-------|----------|-----------|--------|----------|
| **Global resutls** |  **98.20** | **98.23** |  **98.21** |  **98.21** |
| circle | 97.1 | 99.4 | 93.0 | 98.5 |
| triangle | 95.6 | 96.5 | 99.1 | 97.8 |

| ![confusion_matrix.png](https://github.com/SebastianArriagadaS/unsupervised_ml/blob/main/zero-shot/confusion_matrix_tri_crl.png) | 
|:--:| 
| *Confusion matrix* |

In conclusion, this method is highly recommended if it is possible to assign labels that literally describe what is shown in the image. On the other hand, it presents deficient results if the description corresponds to concepts or abstract figures rooted in human collective knowledge.