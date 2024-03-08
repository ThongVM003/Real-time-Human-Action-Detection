import os
import random
import shutil


path = "./data/SLOWFAST"

# Split the data into training and validation sets
split = 0.9


# Create the training and validation sets
for folder in os.listdir(path):
    if folder != ".DS_Store":
        files = os.listdir(os.path.join(path, folder))
        random.shuffle(files)
        split_index = int(len(files) * split)
        train = files[:split_index]
        val = files[split_index:]
        for file in train:
            shutil.move(
                os.path.join(path, folder, file),
                os.path.join(path, "train", folder, file),
            )
        for file in val:
            shutil.move(
                os.path.join(path, folder, file),
                os.path.join(path, "val", folder, file),
            )
        os.rmdir(os.path.join(path, folder))
