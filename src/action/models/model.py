import pathlib
from os.path import join

dir = pathlib.Path(__file__).parent.resolve()


class Models:
    HUMAN_NANO = join(dir, "yolov8n.pt")
    HUMAN_X = join(dir, "yolov8x.pt")
