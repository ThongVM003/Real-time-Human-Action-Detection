format:
	@black .


detect-track-show:
	@python3 src/action/human_tracking.py --source "data/MCF/train/Fall Down/video_0_flip.avi" --show --model_type HUMAN_X

detect-track-save:
	@python3 src/action/human_tracking.py --source "data/MCF/train/Fall Down/video_0_flip.avi" --model_type HUMAN_X --save --conf 0.35

install-env:
	@bash deployment/install.sh

convert:
	@python3 src/action/convert_data.py --source "data/MCF/train/" --dest "data/MCF_UCF24"

train-stgcn:
	@python ST-GCN-Pytorch/train.py