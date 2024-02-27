from ultralytics import YOLO
from rich import print
import numpy as np
import cv2
from utils.track import Detron
from rich import console
from models.model import Models


# Detect human
def detect(
    source,
    net,
    type="image",
    classes=0,
    gpu=True,
    conf=0.3,
    iou=0.45,
    show=False,
    save=False,
):
    """
    Detects objects in an image or video based on ultralytics
    """
    # Set the device
    if gpu:
        net.to("cuda")
    else:
        net.to("cpu")

    # Check the type of source
    if type == "image":
        results = net(source, classes=classes, conf=conf, iou=iou, verbose=False)
        if show:
            imgs = []
            for result in results:
                imgs.append(result.plot())
            # Stack the images
            img = np.hstack(imgs)

    elif type == "video":
        return detect_video(source, net, conf=conf, iou=iou, show=show, save=save)
    else:
        print("[green]Defaulting to ultralytics detect[/green]")
        results = net(source, classes=classes)


def init_net(model_type=Models.HUMAN):
    return YOLO(model_type)


def detect_video(source, net, conf=0.3, iou=0.45, show=False, save=False):
    """
    Detects objects in a video based on ultralytics
    """
    try:
        cap = cv2.VideoCapture(source)
    except:
        print("[red]Error opening video stream or file[/red]")
        return

    detron = Detron("ColorAnnotator")

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    if save:
        out = cv2.VideoWriter(
            "output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    con = console.Console()
    with con.status("[bold green]Processing video...") as status:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect the objects
            results = net(frame, conf=conf, iou=iou, verbose=False)[0]
            detron.add_detection(results)
            frame = detron.track(frame, draw=True)
            if show:
                cv2.imshow("Frame", frame)
            if save:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    print("[green]Done processing video[/green]")
    cap.release()
    cv2.destroyAllWindows()
    if save:
        out.release()
