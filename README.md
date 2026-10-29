# RSOD
## Getting Started
### Installation
```
cd RSOD
conda create -n RSOD python=3.9.24
conda activate RSOD
python -m pip install -r requirements/build.txt
python -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 -c requirements/runtime.txt --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install -U openmim -c requirements/runtime.txt
mim install mmengine==0.10.6 -c requirements/runtime.txt
mim install "mmcv==2.1.0" -c requirements/runtime.txt
python -m pip check
```

### FSOD Dataset

The FSOD dataset contains **3,929 forward-looking sonar images** covering **10 object categories**, with bounding-box annotations in **COCO JSON format**.

Download: [Baidu Netdisk](https://pan.baidu.com/s/1yXjrsPkAfq8GC4Kw44qdjw?pwd=0516) | Extraction code: `0516`

Organize the dataset as follows:

```text
FSOD/
├── train/                          # Training images
├── val/                            # Validation images
├── test/                           # Test images
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    ├── instances_test.json
    ├── instances_10train.json       # 10% labeled training subset
    └── instances_10unlabeled.json   # Remaining unlabeled training subset
```

Image filenames must match the `file_name` entries in the corresponding JSON files.

Set the dataset root in `projects/RSOD/configs/FSOD10_sonar_detection.py`:

```python
data_root = '/path/to/FSOD/'
```

The example training configuration uses `instances_10train.json` and `instances_10unlabeled.json`.

## Train
Run the following sample instructions for training：
```
python tools/train.py projects/RSOD/configs/FSOD10_faster-rcnn_r50_fpn_100_sonar-s1-p10.py
```
## Evaluation
Checkpoints can be evaluated in the following ways：
```
python tools/test.py projects/RSOD/configs/FSOD10_faster-rcnn_r50_fpn_100_sonar-s1-p10.py work_dir/last_checkpoint.pth --out predictions.pkl
```

## Citation

If you find this project useful, please consider citing:
```bash
@inproceedings{li2026rsod,
  title     = {RSOD: Reliability-Guided Sonar Image Object Detection with Extremely Limited Labels},
  author    = {Li, Chengzhou and Guo, Ping and Meng, Guanchen and Jia, Qi and Liu, Jinyuan and Liu, Zhu and Liu, Xiaokang and Liu, Yu and Luo, Zhongxuan and Fan, Xin},
  booktitle = {Proceedings of the {AAAI} Conference on Artificial Intelligence},
  volume    = {40},
  number    = {8},
  pages     = {6055--6063},
  year      = {2026},
  month     = {mar},
  publisher = {{AAAI} Press},
  doi       = {10.1609/aaai.v40i8.37529},
  url       = {https://doi.org/10.1609/aaai.v40i8.37529}
}
```
