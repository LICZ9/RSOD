# RSOD
## Getting Started
### Installation
```
cd RSOD
conda create -n RSOD python=3.9.24
conda activate RSOD
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dataset
Download [FSOD](https://pan.baidu.com/s/10nJoimVv_8gENv-yZyfMEA?pwd=0516) datasets

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
Checkpoints can be evaluated and visualized in the following ways：
```
python tools/test.py projects/RSOD/configs/FSOD10_faster-rcnn_r50_fpn_100_sonar-s1-p10.py work_dir/last_checkpoint.pth --out predictions.pkl
# Get the test result indicators and save the pkl file.

python analyze_results.py projects/RSOD/configs/rsod10_faster-rcnn_r50_fpn_100_sonar-s1-p10.py predictions.pkl visual/
# Visualize the test results and save them in the Visual directory.
```
