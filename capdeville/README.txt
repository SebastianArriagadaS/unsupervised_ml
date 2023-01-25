The goal of this project is to research and experiment with Unsupervised Classification models on the German traffic sign dataset. [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

For my part, I was interested in several models. First, I wanted to see how well a supervised model performed in order to have a performance benchmark.
Then I did some research and tried a model using pre-trained neural networks to extract features from images. I then applied a Kmeans algorithm on these features with or without Principal Component Analysis to predict in an unsupervised way.
In the third part I could test a more advanced model which is the Invariant Information Clustering. I first tested it on the MNIST dataset which is the one on which it performs the best. I then wanted to train it on our dataset but I ran into a non-learning problem that I could not solve.
Finally I did some research on other advanced models with Semantic Clustering. The goal is to transform the images to do Self Supervized as in the previous model and then apply a Nearest Neighbors algorithm. I did not implement it because Sacha was already working on a similar subject.

All my work is detailed on notion. You can read it here https://skinny-dinosaur-7b7.notion.site/J-r-mie-CAPDEVILLE-Report-67f05c7a5f6b488285b732d45c4b10d8. You also have a pdf version but it's not as pretty as notion.
