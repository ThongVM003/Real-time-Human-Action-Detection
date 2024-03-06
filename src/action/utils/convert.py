import os
from rich import print
import cv2
from utils.detect import detect, init_net
from tqdm.rich import tqdm

NET = init_net(model_type="HUMAN_X")


def vid_to_ucf24(video_path, output_path, idx):
    ann_dir = os.path.join(output_path[0], "labels", output_path[1], output_path[2])
    img_dir = os.path.join(output_path[0], "rgb-images", output_path[1], output_path[2])
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    # Read the video
    cnt = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        res = detect(
            frame, NET, type="image", conf=0.35, iou=0.45, show=False, save=False
        )[0]
        # Save the frame as an image in format 00001.jpg
        cv2.imwrite(os.path.join(img_dir, f"{cnt:05d}.jpg"), frame)
        # Save the annotations in format 00001.txt
        with open(os.path.join(ann_dir, f"{cnt:05d}.txt"), "w") as f:
            for bbox in res.boxes.xyxy.cpu().tolist():
                f.write(f"{idx+1} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        cnt += 1


def to_ucf24(source, dest, split=0.1, max_samples=100):
    """
    Converts the UCF-24 dataset from the source directory to the destination directory.

    Args:
        source (str): The path to the source directory or file.
        dest (str): The path to the destination directory.
        split (float, optional): The split ratio for train and test data. Defaults to 0.1.
        max_samples (int, optional): The maximum number of samples to process per class. Defaults to 100.
    """
    # Check if the source is a directory
    if os.path.isdir(source):

        os.makedirs(dest, exist_ok=True)
        os.makedirs(os.path.join(dest, "labels"), exist_ok=True)
        os.makedirs(os.path.join(dest, "rgb-images"), exist_ok=True)
        os.makedirs(os.path.join(dest, "splitfiles"), exist_ok=True)
        # create txt file for train and test
        with open(os.path.join(dest, "splitfiles", "trainlist01.txt"), "w") as f:
            pass
        with open(os.path.join(dest, "splitfiles", "testlist01.txt"), "w") as f:
            pass
        for idx, i in enumerate(os.listdir(source)):
            for k, j in enumerate(
                tqdm(
                    os.listdir(os.path.join(source, i))[:max_samples],
                    desc=f"Processing {i}",
                )
            ):
                if k < max_samples * (1 - split):
                    with open(os.path.join(dest, "trainlist01.txt"), "a") as f:
                        f.write(f"{i}/{j[:-4]}\n")
                else:
                    with open(os.path.join(dest, "testlist01.txt"), "a") as f:
                        f.write(f"{i}/{j[:-4]}\n")
                if j.endswith(".avi") or j.endswith(".mp4"):
                    vid_to_ucf24(os.path.join(source, i, j), (dest, i, j[:-4]), idx)
    elif os.path.isfile(source):
        if source.endswith(".avi") or source.endswith(".mp4"):
            vid_to_ucf24(source, dest + "_ucf24")
        else:
            print("Invalid file format")
    else:
        print("Invalid source path")
    return
