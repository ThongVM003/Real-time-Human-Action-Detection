# REAL TIME HUMAN ACTION DETECTION FOR SUPERVISING IN ENCLOSED-SPACE


## Introduction :cup_with_straw:

An optimize pipeline to apply action detection in real time scenario

## Usage

### Installation :robot:

#### YOWOv2

- We reccomand create a virtual enviroment with:
```
virtualenv venv --python=python3.10
```
- This only works if you have python3.10 installed at the system level (e.g. /usr/bin/python3.10).
- Then activate the enviroment:
```
source venv/bin/activate
```
- Install the requirements:
```Shell
pip install -r requirements.txt
```


#### ST-GCN
- To install the required packages, run the following command:

```bash
make install-env
```

### Dataset
#### MCF-UCF24:
- You can download **MCF-UCF24** from the following link: [Google Drive](https://drive.google.com/file/d/1wUlZ4SnvmCUO-kZwUJ70-WLQKTK5TnIn/view?usp=sharing)

### Human detect and tracking :girl:

To detect and track human in a video, run the following command:

```bash
python3 src/action/main.py --source resources/test/res.mp4 --show
```

## References :star:

## License :book:
