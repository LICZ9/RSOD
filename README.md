# RSOD
## Getting Started
### Installation
```
cd RSOD
conda create -n RSOD python=3.9.24
conda activate RSOD
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -U openmim
mim install mmengine==0.10.6
mim install "mmcv==2.1.0"
```

### Dataset
Download [FSOD](https://pan.baidu.com/s/1p9Vm_Gr8nCVnC7KG6vB0Fg?pwd=0516) datasets
>**Dataset updata (May 2026):** We have released an updatad version of the FSOD dataset with refined annotations. Please use the latest version provided in the download link above.

Please ensure that the dataset meets the following folder structure:

```
$HOME/datasets/
├── FSOD
│ ├── train
│ │ ├── 00001.jpg
│ │ ├── 00002.jpg
...
│ ├── val
│ │ ├── 00001.jpg
│ │ ├── 00002.jpg
...
│ ├── test
│ │ ├── 00001.jpg
│ │ ├── 00002.jpg
...
│ ├── annotations
│ │ ├── instances_train.json
│ │ ├── instances_val.json
│ │ ├── instances_test.json
```
## Train
Run the following sample instructions for training：
```
python tools/train.py projects/RSOD/configs/FSOD10_faster-rcnn_r50_fpn_100_sonar-s1-p10.py
```
## Evaluation
Checkpoints can be evaluated in the following ways：
```
python tools/test.py projects/RSOD/configs/FSOD10_faster-rcnn_r50_fpn_100_sonar-s1-p10.py work_dir/last_checkpoint.pth --out predictions.pkl
# Get the test result indicators 

```
