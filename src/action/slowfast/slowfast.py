import threading
import cv2
import numpy as np
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from typing import Dict
from rich import print, inspect
from PIL import Image

kinetics_id_to_classname = {
    0: "Biking",
    6: "Sitting",
    4: "LyingDown",
    1: "FallDown",
    3: "LongJump",
    8: "Standup",
    7: "Standing",
    2: "Fencing",
    5: "Sitdown",
    9: "Walking",
}


def init_slowfast_model(device):
    # Pick a pretrained model and load the pretrained weights
    model_name = "slowfast_r50"
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", model=model_name, pretrained=True
    )

    # Set to eval mode and move to desired device
    model = model.to(device)
    model = model.eval()

    return model


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        alpha = 4
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def slowfast_transform(video_dict, img_shape=256, num_frames=32, device="cuda"):
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=img_shape),
                CenterCropVideo(crop_size),
                PackPathway(),
            ]
        ),
    )
    inputs = transform(video_dict)["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]
    return inputs


def get_labels(path):
    with open(path, "r") as f:
        kinetics_classnames = json.load(f)
    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")
    return kinetics_id_to_classname


def get_action(inputs, net, kinetics_id_to_classname):
    preds = net(inputs)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=4).indices

    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
    return pred_class_names


def predict_fall(frame_list, net, labels, fallen, bb):
    a = {
        "swinging legs",
        "riding or walking with horse",
        "crawling baby",
        "vault",
        "squat",
        "doing aerobics",
        "abseiling",
        "situp",
        "parkour",
        "stretching leg",
    }
    frames = np.stack(frame_list, axis=1)
    video_tensor = {"video": torch.from_numpy(frames)}
    inputs = slowfast_transform(video_tensor)
    pred_class_names = get_action(inputs, net, labels)
    if a.intersection(set(pred_class_names)):
        fallen.append((pred_class_names, bb))


def detect_fall(frame, frame_queue, net, labels):
    fallen = []
    threads = []
    while not frame_queue.empty():
        print("Detecting fall")
        id = frame_queue.get()
        frames = id[0]
        tmp = threading.Thread(
            target=predict_fall, args=(frames, net, labels, fallen, id[1])
        )
        threads.append(tmp)
        tmp.start()

    for t in threads:
        t.join()

    # draw bounding box
    for fall in fallen:
        print("#" * 20)
        print("Fall detected", fall[0])

        bb = fall[1].astype(int)
        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"Fall detected",
            (bb[0], bb[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )
    return frame, "Detected fall" if fallen else None


if __name__ == "__main__":
    # from utils_funcs import save_frames

    # video_tensor = {
    #     "video": save_frames("./vid_fall_1695285068.mp4"),
    # }

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # net = init_slowfast_model(device)
    # net = net.to(device)

    kinetics_id_to_classname = get_labels("kinetics_classnames.json")
    print(kinetics_id_to_classname.values())
    # Apply the transform to see the result
    # inputs = slowfast_transform(video_tensor, device=device)

    # pred_class_names = get_action(inputs, net, kinetics_id_to_classname)
    # print("Predicted labels:", *pred_class_names)
