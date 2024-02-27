from typing import Any
import supervision as sv
from utils.functions import *
from rich import print


SIMILAR_THRESHOLD = 0.9


class Detron:
    def __init__(self, annotator_type):
        self.box_annotator = self.set_annotators(annotator_type)
        self.tracker = sv.ByteTrack()
        self.label_annotator = sv.LabelAnnotator()

    def set_annotators(self, annotator_type):
        accepted_types = [
            "BoundingBoxAnnotator",
            "RoundBoxAnnotator",
            "BoxCornerAnnotator",
            "ColorAnnotator",
            "CircleAnnotator",
            "DotAnnotator",
            "TriangleAnnotator",
            "EllipseAnnotator",
            "HaloAnnotator",
            "PercentageBarAnnotator",
            "BlurAnnotator",
            "PixelateAnnotator",
        ]

        for type in accepted_types:
            simili = check_str_similarity(annotator_type.lower(), type.lower())
            if simili / len(annotator_type) > SIMILAR_THRESHOLD:
                print(f"Setting annotator to [bold blue]{type}[/bold blue]")
                return eval(f"sv.{type}()")

        print(
            f"Invalid annotator type: {annotator_type}. Setting to default [bold blue]BoundingBoxAnnotator[/bold blue]"
        )
        return sv.BoundingBoxAnnotator()

    def draw_annotations(self, detections, frame, labels=None):
        if type(detections) != sv.Detections:
            detections = self.add_detection(detections)

        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        if labels:
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
        return annotated_frame

    def track(self, frame, detections=None, draw=True):
        if detections is None:
            detections = self.detections
        detections = self.tracker.update_with_detections(detections=detections)
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        frame = self.draw_annotations(detections, frame, labels)
        return frame

    def __call__(self, detections):
        return self.add_detection(detections)

    def add_detection(self, detections):
        self.detections = sv.Detections.from_ultralytics(detections)
        return self.detections
