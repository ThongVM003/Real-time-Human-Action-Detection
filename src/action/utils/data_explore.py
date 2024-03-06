import os
from os.path import isfile, join
import pandas as pd
from rich import print


def ucf24(dir):
    labels = []
    for i in os.listdir(join(dir, "labels")):
        labels.append((i, len(os.listdir(join(dir, "labels", i)))))
        # print(i, ":", len(os.listdir(join(dir, "labels", i))))
    images = []
    for i in os.listdir(join(dir, "rgb_images")):
        images.append((i, len(os.listdir(join(dir, "rgb_images", i)))))
        # print(i, ":", len(os.listdir(join(dir, "images", i))))
    print("Labels", labels)
    print("Images", images)


def AVA(dir):
    classes = []
    with open(dir + "/ava_action_list_v2.2.pbtxt", "r") as f:
        for i in f:
            if "name" in i:
                classes.append(i.split('"')[1])
    df = pd.read_csv(dir + "/ava_train_v2.2.csv", header=None)
    classes_num = df[6].value_counts()
    for i in range(len(classes)):
        print(classes[i], ":", classes_num[i + 1])


def MCF(dir):
    for i in os.listdir(join(dir, "train")):
        print(i, ":", len(os.listdir(join(dir, "train", i))))
