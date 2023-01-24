from transformers import pipeline
from tqdm.auto import tqdm
import os
import pandas as pd
import random
import argparse

# get current directory
path = os.getcwd()
parent_path = os.path.dirname(path)

def process_images(labels_num, n_images):
    # load txt with the labels and put it in a list
    with open(f"{path}/labels.txt", "r") as f:
        labels = f.read().splitlines()

    print(f"Labels ({len(labels)}):")
    for i, label in enumerate(labels):
        print(f"{i}: {label}")

    print(f"Labels to process ({len(labels_num)}):")
    for i, label in enumerate(labels_num):
        print(f"{i}: {labels[label]}")

    for num in labels_num:
        print(f"Loading images from label {num} - {labels[num]}")
        img_path = f"{parent_path}/data/train/{num}/"
        images_names = os.listdir(img_path)
        # take 100 random images
        images_names = random.sample(images_names, n_images)
        # add the url to the image name
        images = [img_path + image_name for image_name in images_names]
        # print(len(images), images)
        results = {}
        # use tqmd to show the progress and the number of images
        preds = vision_classifier(
            images=images,
            candidate_labels=labels,
            hypothesis_template="This is an image of a {} road sign.",
        )
        i = 0
        for pred in tqdm(preds, desc="Images", total=len(images)):
            # sort the labels by score. the preds is a dict with the labels and the scores
            pred = sorted(pred, key=lambda x: x["score"], reverse=True)
            # store all the predictions
            results[images[i]] = {}
            for j, p in enumerate(pred):
                label_text = f"label_{j+1}"
                score_text = f"score_{j+1}"
                results[images[i]][label_text] = p["label"]
                results[images[i]][score_text] = p["score"]
            # results[images[i]] = {
            #     "label_1st": pred[0]["label"],
            #     "label_2nd": pred[1]["label"],
            #     "label_3rd": pred[2]["label"],
            #     "scores_1st": pred[0]["score"],
            #     "scores_2nd": pred[1]["score"],
            #     "scores_3rd": pred[2]["score"],
            # }
            i += 1

        # put the results in a dataframe
        df = pd.DataFrame(results).T
        print(df.head())

        # save the results in a csv
        df.to_csv(f"{path}/results/results_{num}.csv")
        print(f"Results saved in {path}/results/results_{num}.csv")

        # count how many times appear the num label
        print("Correct label:", labels[num])
        print("Accurate:", df["label_1"].value_counts() / len(df) * 100)


# main
if __name__ == "__main__":
    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--labels_num",
        nargs="*",
        # type int or list of int
        type=int,
        default=-1,
        help="List of the labels to process. If not specified, all the labels will be processed",
    )
    parser.add_argument(
        "-n",
        "--n_images",
        type=int,
        default=100,
        help="Number of images to process per label.",
    )
    args = parser.parse_args()
    n_images = args.n_images
    labels_num = args.labels_num
    # if labels_num is not specified, process all the labels
    print(labels_num)
    if labels_num == -1:
        labels_num = range(43)
    

    vision_classifier = pipeline(task="zero-shot-image-classification")
    process_images(labels_num, n_images)
