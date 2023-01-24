import os
import pandas as pd

delete_underscore = False

# get current directory
path = os.getcwd()
parent_path = os.path.dirname(path)
results_folder = "results_2"
# load the results csv
result_list = os.listdir(f"{path}/{results_folder}")

# load the labels
with open(f"{path}/labels.txt", "r") as f:
    labels = f.read().splitlines()

if delete_underscore:
    labels = [label.replace("_", " ") for label in labels]

matrix = {}
for label in labels:
    matrix[label] = {}
    for label2 in labels:
        matrix[label][label2] = 0

for result in result_list:
    df = pd.read_csv(f"{path}/{results_folder}/{result}")
    correct_label = labels[int(result.split("_")[1].split(".")[0])]
    # count the unique labels in the first column "label_1"
    predictions = df["label_1"].value_counts()
    # add the results to the matrix
    for label in predictions.index:
        matrix[label][correct_label] = predictions[label]

# create the confusion matrix
df = pd.DataFrame(matrix)
df.to_csv(f"{path}/confusion_matrix.csv", index=True)

# plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
plot = False
# create the plot
if plot:
    plt.figure(figsize=(20, 20))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"{path}/confusion_matrix.png")
    plt.show()

# calculate the confusion matrix accuracy, precision, recall and f1 score per label
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# get the predictions and the true labels
predictions = []
true_labels = []
for result in result_list:
    df = pd.read_csv(f"{path}{results_folder}/{result}")
    correct_label = labels[int(result.split("_")[1].split(".")[0])]
    # get the predictions
    predictions += df["label_1"].tolist()
    # get the true labels
    true_labels += [correct_label] * len(df)


# calculate the confusion matrix
cm = confusion_matrix(true_labels, predictions, labels=labels)
# calculate the accuracy, precision, recall and f1 score
acc = accuracy_score(true_labels, predictions)
prec = precision_score(true_labels, predictions, average="weighted")
rec = recall_score(true_labels, predictions, average="weighted")
f1 = f1_score(true_labels, predictions, average="weighted")

# print the results
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 Score: {f1}")

# go label by label and calculate the accuracy, precision, recall and f1 score
# calculate the TP, FP, FN and TN
TP = {}
FP = {}
FN = {}
TN = {}
for label in labels:
    TP[label] = 0
    FP[label] = 0
    FN[label] = 0
    TN[label] = 0

for i in range(len(predictions)):
    if predictions[i] == true_labels[i]:
        TP[true_labels[i]] += 1
    else:
        FP[predictions[i]] += 1
        FN[true_labels[i]] += 1

for label in labels:
    for label2 in labels:
        if label != label2:
            TN[label] += cm[labels.index(label)][labels.index(label2)]

# calculate the accuracy, precision, recall and f1 score
signle_results = {}
for label in labels:
    signle_results[label] = {}
    signle_results[label]["accuracy"] = (TP[label] + TN[label]) / (TP[label] + TN[label] + FP[label] + FN[label])
    try:
        signle_results[label]["precision"] = TP[label] / (TP[label] + FP[label])
    except ZeroDivisionError:
        signle_results[label]["precision"] = 0
    signle_results[label]["recall"] = TP[label] / (TP[label] + FN[label])
    try:
        signle_results[label]["f1_score"] = 2 * signle_results[label]["precision"] * signle_results[label]["recall"] / (signle_results[label]["precision"] + signle_results[label]["recall"])
    except ZeroDivisionError:
        signle_results[label]["f1_score"] = 0
# print the results
print("Results per label:")
for label in labels:
    print(f"{label}:")
    print(f"Accuracy: {signle_results[label]['accuracy']}")
    print(f"Precision: {signle_results[label]['precision']}")
    print(f"Recall: {signle_results[label]['recall']}")
    print(f"F1 Score: {signle_results[label]['f1_score']}")
    print()

# create a csv with the results per label
df = pd.DataFrame(signle_results).T
df.to_csv(f"{path}/results_per_label.csv", index=True)




