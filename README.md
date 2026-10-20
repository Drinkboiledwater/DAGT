# No-Reference Derained Image Quality Assessment: From Subjective Benchmarking to Distortion-Aware Representation Learning

Official implementation of No-Reference Derained Image Quality Assessment: From Subjective Benchmarking to Distortion-Aware Representation Learning. Our DQA software and dataset are available at [here](https://pan.baidu.com/s/1Dkch8cDRR82GOmPgDP0ASw?pwd=qmef).

## Highlights

- **RCTB (Refined Convolutional Vision Transformer Block)**  
  Each stage is `FRM(CvT_Stage(...))`: the pretrained CvT stage provides Conv Token Embedding + Transformer Block, and a lightweight **Feature Refinement Module (FRM)** restores sensitivity to subtle local degradations.
- **RDE (Rain Distortion Extractor)**  
  Encoder–decoder network producing a **384-channel** rain-distortion representation (used as RAAM query).
- **SDE (Structural Distortion Extractor)**  
  Sobel + StemConv + Residual + AttnConv pipeline producing a **512-channel** structural-distortion representation (used as SAAM query).
- **RAAM / SAAM (Distortion-guided Feature Enhancement)**  
  Unified cross-attention module (4 heads). RAAM pairs the RDE prior with CvT stage0 (`F_1`), SAAM pairs the SDE prior with CvT stage2 (`F_3`), producing `F_r` and `F_s`.
- **Ranking-aware regression**  
  Training loss combines MSE with a pairwise ranking loss (`0.7 * MSE + 0.3 * Rank`).
- **Pre-extracted features**  
  CvT stages, RDE, and SDE are pre-computed once per image, so training only runs the lightweight regression head.

## Repository layout


```
.
├── cvt.py # CvT backbone (stage0 / stage1 / stage2)
├── models.py # RDE, SDE, UNetGenerator, PatchGANDiscriminator
├── main.py # FeatureRefinementModule, CvTFeatureExtractor,
│ # DistortionGuidedEnhancement (RAAM / SAAM),
│ # CvTFeatureRegression, CvTIQATrainer
├── pre_load.py # Pre-extract CvT + RDE + SDE features to *.pt
├── split_scenes.py # Scene-level train/val/test split (7:1:2)
├── make_dummy_ckpts.py # Generate random RDE/SDE weights (smoke test)
├── requirements.txt
├── checkpoint/
│ ├── CvT-21-384x384-IN-22k.pth # pretrained CvT (required)
│ ├── rde_best.pth # trained RDE (required)
│ └── sde_best.pth # trained SDE (required)
└── preload_derain1_scene/
├── train39/ # pre-extracted training features
├── val39/
└── test39/
```

## Environment

Tested on Python 3.9 + PyTorch with CUDA.

```bash
conda create -n cvt-iqa python=3.9 -y
conda activate cvt-iqa
pip install -r requirements.txt
```

A GPU with ≥ 8 GB memory is recommended for `pre_load.py` (CvT-21 at 384×384 + RDE + SDE run simultaneously).

## Data preparation

### 1. Label CSV

`mos1.csv` with two columns:

```csv
picture,mos
0001_Fu2019.png,0.8421
0001_Jiang2020.png,0.7630
...
```

### 2. Scene-level 7:1:2 split

```bash
python split_scenes.py
# -> train39.csv / val39.csv / test39.csv
```

The split is done **at scene level** so images of the same scene never leak across splits. Edit `csv_path` and `random_state` at the top of `split_scenes.py` if needed.

### 3. Pre-extract features

```bash
# smoke test (3 samples)
python pre_load.py --split val --limit 3

# full extraction
python pre_load.py
```

Each `.pt` file contains:

| key          | shape               | source                          |
|--------------|---------------------|---------------------------------|
| `f0`         | `[1, 64, 96, 96]`   | CvT stage0 feature map          |
| `f1`         | `[1, 192, 48, 48]`  | CvT stage1 feature map          |
| `f2`         | `[1, 384, 24, 24]`  | CvT stage2 feature map          |
| `cls`        | `[1, 1, 384]`       | CvT stage2 CLS token            |
| `rain`       | `[1, 384, 6, 6]`    | RDE output (adaptive pooled)    |
| `blur`       | `[1, 512, 6, 6]`    | SDE output (adaptive pooled)    |
| `mos`        | scalar              | ground-truth MOS                |
| `image_name` | str                 | original filename               |

Approximate disk usage: **~4.9 MB per image** with `PRIOR_SIZE=(6, 6)`.

If RDE / SDE are trained on a different input resolution, change `transforms.Resize((256, 256))` inside `CvTFeatureExtractor.load_image` before running `pre_load.py`.

## Training

```bash
python main.py
```

Default configuration (see `main()` in `main.py`):

| item            | value                                       |
|-----------------|---------------------------------------------|
| optimizer       | AdamW (`lr=1e-4`, `weight_decay=1e-4`)      |
| scheduler       | CosineAnnealingLR (`T_max=50`)              |
| batch size      | 16                                          |
| epochs          | 150                                          |
| loss            | `0.7 * MSE + 0.3 * RankingLoss`             |
| checkpoint rule | save when **val PLCC** improves             |

Outputs:
- `bestplcc.pth` – best regression head
- `training_results.png` – scatter + error histogram on the test split

## Evaluation

Metrics reported on the test split: **MSE / MAE / RMSE / PLCC / SROCC / KROCC**.

```python
from main import CvTIQATrainer, DerainIQADataset, CvTFeatureRegression
from torch.utils.data import DataLoader

# build loaders as in main(), then:
trainer = CvTIQATrainer(model, train_loader, val_loader, test_loader)
trainer.test("bestplcc.pth")
```

## Citation

If you find this work useful, please cite our paper.

## Acknowledgements

- CvT backbone adapted from [Wu et al., 2021](https://arxiv.org/abs/2103.15808).

