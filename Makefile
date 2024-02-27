format:
	@black .


detect-track-show:
	@python3 src/action/main.py --source resources/test/res.mp4 --show

detect-track-save:
	@python3 src/action/main.py --source resources/test/res.mp4 --save

install-env:
	@bash deployment/install.sh