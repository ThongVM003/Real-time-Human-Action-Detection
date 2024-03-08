from src.action.utils.data_explore import ucf24, AVA, MCF
import os
from os.path import join
from rich import print
import shutil

# ucf24("./data/MCF_UCF24")

with open("./data/MCF_UCF24/splitfiles/trainlist01.txt") as f:
    videos_folder_train = f.readlines()
    videos_folder_train = [v.strip() for v in videos_folder_train]

with open("./data/MCF_UCF24/splitfiles/testlist01.txt") as f:
    videos_folder_test = f.readlines()
    videos_folder_test = [v.strip() for v in videos_folder_test]

DIR = "./data/MCF_UCF24/rgb_images"
for i in os.listdir(DIR):
    img = 0
    for j in os.listdir(join(DIR, i)):
        if (
            not join(i, j) in videos_folder_train
            and not join(i, j) in videos_folder_test
        ):
            # Delete the folder
            shutil.rmtree(join(DIR, i, j))
            print("Deleted", join(DIR, i, j))
            # exit()
        else:
            img += 1
    print(i, img)
