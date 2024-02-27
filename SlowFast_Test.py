import torch
import json
import urllib
import cv2
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
    UniformCropVideo
)

def load_model(device="cpu"):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    model = model.eval().to(device)
    return model

def load_kinetics_classnames(json_url):
    json_filename = "kinetics_classnames.json"
    try:
        urllib.URLopener().retrieve(json_url, json_filename)
    except:
        urllib.request.urlretrieve(json_url, json_filename)
    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)
    kinetics_id_to_classname = {v: str(k).replace('"', "") for k, v in kinetics_classnames.items()}
    return kinetics_id_to_classname

class PackPathway(torch.nn.Module):
    def __init__(self, slowfast_alpha=4):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha
        
    def forward(self, frames):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

def get_transform(side_size=256, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], crop_size=256, num_frames=32, sampling_rate=2):
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size),
            PackPathway(),
        ])
    )
    return transform

def predict_action(model, video_data, device):
    with torch.no_grad():
        inputs = video_data["video"]
        inputs = [i.to(device)[None, ...] for i in inputs]
        preds = model(inputs)
    return preds

def visualize_prediction(video_path, pred_class_names, confidence_scores, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label_text = ", ".join([f"{label} ({score:.2f})" for label, score in zip(pred_class_names, confidence_scores)])
        cv2.putText(frame, label_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        cv2.imshow('Video with Action Recognition using SlowFast Networks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(device)

    kinetics_id_to_classname = load_kinetics_classnames("https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json")
    transform = get_transform()
    
    clip_duration = (32 * 2) / 30
    start_sec = 0
    end_sec = start_sec + clip_duration
    video_path = './Downloads/365408122_838145721356263_6642055713205857660_n.mp4'

    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    video_data = transform(video_data)
    preds = predict_action(model, video_data, device)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=1).indices[0]
    confidence_scores = preds.max(dim=1).values 
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    visualize_prediction(video_path, pred_class_names, confidence_scores, './Downloads/output_video_2.mp4')
