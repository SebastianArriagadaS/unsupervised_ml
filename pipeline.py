from transformers import pipeline
from tqdm.auto import tqdm
import os
import pandas as pd
import random
import argparse


def process_images(labels_num, n_images):
    # load txt with the labels and put it in a list
    with open("data/labels.txt", "r") as f:
        labels = f.read().splitlines()
    print(f"Labels {len(labels)}: {labels}")
    for num in labels_num:
        print(f"Loading images from folder {num}")
        url = f"data/train/{num}/"
        images_names = os.listdir(url)
        # take 100 random images
        images_names = random.sample(images_names, n_images)
        # add the url to the image name
        images = [url + image_name for image_name in images_names]
        # print(len(images), images)
        results = {}
        # use tqmd to show the progress and the number of images
        preds = vision_classifier(
            images=images,
            candidate_labels=labels,
            hypothesis_template="This image is a picture of a {} traffic sign.",
        )
        i = 0
        for pred in tqdm(preds, desc="Images", total=len(images)):
            # sort the labels by score. the preds is a dict with the labels and the scores
            pred = sorted(pred, key=lambda x: x["score"], reverse=True)
            # store the three best predictions
            results[images[i]] = {
                "label_1st": pred[0]["label"],
                "label_2nd": pred[1]["label"],
                "label_3rd": pred[2]["label"],
                "scores_1st": pred[0]["score"],
                "scores_2nd": pred[1]["score"],
                "scores_3rd": pred[2]["score"],
            }
            i += 1

        # put the results in a dataframe

        df = pd.DataFrame(results).T
        print(df.head())

        # save the results in a csv
        df.to_csv(f"results/results_{num}.csv")
        print(f"Results saved in results/results_{num}.csv")

        # count how many times appear the num label
        print("Correct label:", labels[num])
        print(df["label_1st"].value_counts())


# main
if __name__ == "__main__":
    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--labels_num",
        nargs="*",
        type=list,
        default=[1, 2],
        help="List of the labels to process",
    )
    parser.add_argument(
        "-n",
        "--n_images",
        type=int,
        default=100,
        help="Number of images to process",
    )
    args = parser.parse_args()
    n_images = args.n_images
    labels_num = args.labels_num
    # check if the labels has lists inside
    if isinstance(labels_num[0], list):
        labels_num = [item for sublist in labels_num for item in sublist]
    vision_classifier = pipeline(task="zero-shot-image-classification")
    process_images(labels_num, n_images)
