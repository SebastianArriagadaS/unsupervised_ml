# Read the csv files in results folder and count how many times appear the correct label

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# get current directory
path = os.getcwd()
parent_path = os.path.dirname(path)
folder_name = "results_2"
delete_label_underscore = False
# load the labels
with open(f"{path}/labels.txt", "r") as f:
    labels = f.read().splitlines()

# load the results
results = pd.DataFrame()
for file in os.listdir(f"{path}/{folder_name}"):
    # load the csv
    df = pd.read_csv(f"{path}/{folder_name}/{file}")
    # add the correct label
    df["correct_label"] = labels[int(file.split("_")[1].split(".")[0])]
    # add the results to the results dataframe
    results = pd.concat([results, df])

# chanche the underscore to space in the elements of the columns label_1st, label_2nd, label_3rd
if delete_label_underscore:
    results["label_1"] = results["label_1"].str.replace("_", " ")
    results["label_2"] = results["label_2"].str.replace("_", " ")
    results["label_3"] = results["label_3"].str.replace("_", " ")

# count how many times the correct label appears in the first, second and third position
results["1st"] = results["correct_label"] == results["label_1"]
results["2nd"] = results["correct_label"] == results["label_2"]
results["3rd"] = results["correct_label"] == results["label_3"]

# sum the results
acc_1st = results["1st"].sum()
acc_2nd = results["2nd"].sum()
acc_3rd = results["3rd"].sum()

# get the accumulated results
acc_3rd = (acc_1st + acc_2nd + acc_3rd) / len(results)
acc_2nd = (acc_1st + acc_2nd) / len(results)
acc_1st = acc_1st / len(results)

# plot the results in a barplot
# sns.barplot(x=["1st", "2nd", "3rd"], y=[acc_1st, acc_2nd, acc_3rd])
# plt.title("Accuracy of the predictions")
# plt.xlabel("Position")
# plt.ylabel("Accuracy")
# plt.show()

# create a plot per present label
present_labels = results["correct_label"].unique()
# to save the results
data_dict = {}
for label in labels:
    if label not in present_labels:
        continue
    # get the results for the label
    df = results[results["correct_label"] == label].copy()
    # count how many times the correct label appears in the first, second and third position
    df["1st"] = df["correct_label"] == df["label_1"]
    df["2nd"] = df["correct_label"] == df["label_2"]
    df["3rd"] = df["correct_label"] == df["label_3"]
    # sum the results
    acc_1st = df["1st"].sum()
    acc_2nd = df["2nd"].sum()
    acc_3rd = df["3rd"].sum()
    # plot the results in a barplot
    sns.barplot(
        x=["1st", "2nd", "3rd"],
        y=[acc_1st, acc_2nd + acc_1st, acc_3rd + acc_2nd + acc_1st],
    )
    plt.title(f"Acummulative accuracy of the predictions for label {label}")
    plt.xlabel("Position")
    plt.ylabel("Accuracy")
    # set the y axis from 0 to 100
    plt.ylim(0, 100)
    txt = f"{label} 1st: {acc_1st} 2nd: {acc_2nd} 3rd: {acc_3rd}"
    # save the results
    data_dict[label] = [acc_1st, acc_2nd, acc_3rd, acc_1st + acc_2nd + acc_3rd]
    print(txt)
    # plt.show()

# save the results in a csv file
df = pd.DataFrame.from_dict(
    data_dict, orient="index", columns=["1st", "2nd", "3rd", "total"]
)
df.to_csv("summary_results.csv")
print("Results saved in summary_results.csv")

# print the results
print(df)
