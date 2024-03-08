path = (
    "/workspaces/rtsptoweb/Real-time-Human-Action-Detection/data/MCF_UCF24/rgb-images"
)


out_path = "/workspaces/rtsptoweb/Real-time-Human-Action-Detection/data/SLOWFAST/"
import os
import cv2
from tqdm.auto import tqdm
import multiprocessing as mp

split = 0.9

# Create the training and validation sets
os.makedirs(os.path.join(out_path, "train"), exist_ok=True)
os.makedirs(os.path.join(out_path, "val"), exist_ok=True)


def do(path, out_path, folder, ayo):
    for idx, vid in enumerate(
        tqdm(
            sorted(os.listdir(os.path.join(path, folder))),
            desc=folder,
            colour="green",
            position=ayo,
            leave=True,
        )
    ):
        imgs = sorted(os.listdir(os.path.join(path, folder, vid)))
        if len(imgs) < 30:
            continue
        if idx < 0.9 * len(os.listdir(os.path.join(path, folder))):
            os.makedirs(os.path.join(out_path, "train", folder), exist_ok=True)
            out = cv2.VideoWriter(
                os.path.join(out_path, "train", folder, vid + ".avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                25,
                (256, 256),
            )
        else:
            os.makedirs(os.path.join(out_path, "val", folder), exist_ok=True)
            out = cv2.VideoWriter(
                os.path.join(out_path, "val", folder, vid + ".avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                30,
                (256, 256),
            )

        for img in imgs:
            image = cv2.imread(os.path.join(path, folder, vid, img))
            image = cv2.resize(image, (256, 256))
            # print(image.shape)
            # break
            out.write(image)
        out.release()


if __name__ == "__main__":
    with mp.Pool(10) as p:
        p.starmap(
            do,
            [
                (path, out_path, folder, ayo)
                for ayo, folder in enumerate(sorted(os.listdir(path)))
            ],
        )
