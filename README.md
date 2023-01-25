# ADND-J7-unsupervised-ml

This repository contains the code for the unsupervised machine learning project of the Advanced Deep Neural Networks course at the University of Eotvos Lorand, Budapest.

This project is based on the German traffic sign recognition benchmark dataset. The dataset contains 43 different classes of traffic signs. The goal of the project is find the best unsupervised learning method to cluster the traffic signs into the 43 classes.

## Jérémie's work : 

For my part, I was interested in several models. First, I wanted to see how well a supervised model performed in order to have a performance benchmark.
Then I did some research and tried a model using pre-trained neural networks to extract features from images. I then applied a Kmeans algorithm on these features with or without Principal Component Analysis to predict in an unsupervised way.
In the third part I could test a more advanced model which is the Invariant Information Clustering. I first tested it on the MNIST dataset which is the one on which it performs the best. I then wanted to train it on our dataset but I ran into a non-learning problem that I could not solve.
Finally I did some research on other advanced models with Semantic Clustering. The goal is to transform the images to do Self Supervized as in the previous model and then apply a Nearest Neighbors algorithm. I did not implement it because Sacha was already working on a similar subject.

![Untitled](https://user-images.githubusercontent.com/116168201/214251296-86fea8b4-c3ab-4f5a-b4e1-8dcef83821cf.png)

## Sacha's work : 

I chose to focus on K-NN model approach. This let us have a comparison between Supervised and Unsupervised Learning models and check if there is a big improve or not. The algorithms details and principles are detailled in my report part. I did i fact 4 different approachs : 
- A Standard KNN approach by coputing all our distances by flattening our pictures. (accuracy ~40%)
- A PCA + K-NN in order to reduce our dimension (accuracy ~7%)
- A Semi-Supervised Label Propagation with knn kernel approach (accuracy ~35%)
- A Neural Network model (ResNet50) (accuracy ~70%)
- A ResNet50 + Dense-KNN layer (accuracy ~2%)

K-NN is a really good Supervised Learning model for numeric data in small dimensions, or data that can be reduced easily, because our pictures represent too much data for a "simple" Machine Learning algorithm that can't handle really high-dimension spaces (our was more than 3000). The Neural Network was really useful and has great performances.

## Mei Jiaojiao's work :

The whole repository can be found at the following link:
https://github.com/JIAOJIAOMEI/ImageClustering_J7project

This is jiaojiao's independent work for J7project, the deep clustering task of traffic signs.

One can review the summary of literature and also the experiments at the following link(**references of literature and code are also attached in the end**):

https://github.com/JIAOJIAOMEI/ImageClustering_J7project/blob/main/Summary/summary_v6fisc.md

here is a pdf version of the summary with **table of contents**, please find it at the link:

https://github.com/JIAOJIAOMEI/ImageClustering_J7project/blob/main/Summary/summary_v6fisc.pdf

One can find the code and the dataset in this file:

https://github.com/JIAOJIAOMEI/ImageClustering_J7project/tree/main/Code

**Before get into the details, here is a short descrption of my work**:

For machine learning methods, I tried 1. pure K-means; 2.transfer learning+K-means; 3.Agglomerative clustering + PCA with 4 different linkages.

For deep learning methods, I tried several ways to implement autoencoders, which are 1.simple autoencoders; 2.Multiplayer auto encoders; 3. Convolutional autoencoders and 4. Regularized autoencoders.

- Pure K-means received the accuracy at 18%.
- Transfer learning does not improve pure K-means, the pertained model I used is ResNet50, this model is trained on ImageNet which contains most natrual images, not by our case (traffic signs). Although it does not improve the accuracy, it does not decrease it either.
- I tried single linkage, ward linkage, centroid linkage and complete linkage when using Agglomerative clustering + PCA method. I think ward linkage performs the best outcome between these 4, and better than pure Kmeans, also better than transfer learning. The evidence is that the 43 centers produced by ward linkage are most different from each other.
- Among all the 4 autoencoders I tried, I think the performance are organized as the following: multilayer autoencoders>simple autoencoders>regularized autoencoders>convolutional autoencoders. The convolutional auto encoders I tried cannot give any useful information at all, maybe the training data is not enough, or maybe somewhere I made a mistake. The convolutional autoencoders is the most complicated autoencoders, but it fails to meet my expectation.
- Deep learning methods are not necessarily better than machine learning methods in a specific situation, it depends on many factors, for example, the coding ability of who implements it, or GPU resources, or the level of feature extraction, or the understanding of the task.

The reason why I choose K-means and Agglomerative clustering, is simply because that they produce results like "one picture can only be in one class", this is very important. Our dataset is about traffic signs, these classes are strongly independent, either it belongs to this category, or it belongs to another category. Another reason is that these methods are easily to be implemented.

Although the classes/categories are strongly independent, the pictures are not that kind of independent. For example, the following pictures do have something in common. This is what makes the task difficult. Machine learning methods can learn the shapes and the colors, but fails to learn the details like "2","3","5", and something like this.

![43 categories](https://github.com/JIAOJIAOMEI/ImageClustering_J7project/blob/main/43%20categories.png)

The reason why I choose autoencoders, is also simple. There are 3 ways: 1. Feed-forward networks; 2. autoencoders (a type of feed-forward networks designed for clustering); 3. GAN & VAE. It seems like two ways are about feed-forward networks, and autoencoders are designed for this. Then, I tried autoenders. I think GAN & VAE are also very interesting, maybe next time I will try. 

## Sebastian Arriagada's work :

The chosen unsupervised method was [Zero-Shot Image Classification](https://github.com/SebastianArriagadaS/unsupervised_ml/tree/main/zero-shot), which is a subgroup of transfer learning. 

The CLIP Vit model was used, which consists of a text transformer for encoding text embeddings and a vision transformer (ViT) for encoding image embeddings, allowing for the association of an image with potential labels provided. 

It can be concluded that this model is quite sensitive to the provided labels and the similarity of new images to the training database. As seen in the results, there are explicit labels regarding the image where very high **F1 scores** are obtained, such as the stop sign with **92%** or speed limit signs with ranges between **87 to 50%**. However, for labels with names that do not exactly describe what the image shows, such as "End of no overtaking zone" the **F1 score** is close to **0%**.

An extract of the results is presented below.

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
| Ahead or turn right only | 44.6 | 0.0 | 0.0 | 0.0 |
| Ahead or turn left only | 42.6 | 0.0 | 0.0 | 0.0 |
| Bypass on right | 39.4 | 0.0 | 0.0 | 0.0 |
| Bypass on left | 40.3 | 10.0 | 6.0 | 7.5 |
| Roundabout | 35.7 | 16.7 | 20.0 | 18.2 |
| End of no overtaking zone | 37.7 | 1.5 | 1.0 | 1.2 |
| End of no overtaking zone for trucks | 44.6 | 0.0 | 0.0 | 0.0 |

If we focus on the [confusion matrix](https://github.com/SebastianArriagadaS/unsupervised_ml/tree/main/zero-shot/confusion_matrix.png), we can observe that most errors occur in False Positives with the label "General Warning", this can be due to the model having learned the general concept of a traffic sign. Along with this, it is important to consider that there are two types of patterns that repeat in all signs, which are the circular and triangular shapes. This can make recognition more difficult, as to differentiate them, the model has to focus mainly on the center area.

Therefore, this method may be more useful for research on general features, for example, the results presented below show that if we only differentiate between triangular and circular signs, we obtain very good results.

| Label | Accuracy | Precision | Recall | F1_score |
|-------|----------|-----------|--------|----------|
| **Global resutls** |  **98.20** | **98.23** |  **98.21** |  **98.21** |
| circle | 97.1 | 99.4 | 93.0 | 98.5 |
| triangle | 95.6 | 96.5 | 99.1 | 97.8 |