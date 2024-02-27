import pathlib
from os.path import join

dir = pathlib.Path(__file__).parent.resolve()


class Models:
    HUMAN = join(dir, "yolov8n.pt")
