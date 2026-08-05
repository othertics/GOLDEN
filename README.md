# GOLDEN
> **GOLDEN**: **G**ene expressi**O**n and a**L**lele-informed **DE**composition for ge**N**etic heterogeneity

**GOLDEN** is a deep learning framework developed to decipher inter-individual heterogeneity in spatial transcriptomics (ST).

---

## Environment Setup

We recommend running GOLDEN in the following environment (tested on **Linux Rocky 8.10**):

- **python**: 3.10
- **torch**: 2.9
- **numpy**: 2.2
- **matplotlib**: 3.10

### Installation
```bash
# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate golden
```

## Data & Model Weights Download
| Item | Description | Link |
|---|---|---|
| Data | GEO dataset | [GSE341398](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE341398) |
| Model Weights | Trained model checkpoint | [golden.ckpt](https://huggingface.co/datasets/othertics/golden/golden.ckpt) |
| Simulation Sample | Simulation data for test | [test_data](https://huggingface.co/datasets/othertics/golden/test_data) |

---

## 1. Preprocessing

Use `preprocessing.py` for preprocessing.

```
usage: preprocessing.py [-h] --data-root DATA_ROOT --geneinfo GENEINFO --out-root OUT_ROOT
                        [--save-formats {binary,raw,cp10k_log1p} ...] [--no-aac-features]
                        [--table-format {csv,tsv}] [-v]
```

### Options
 
| Option | Description |
|---|---|
| `-h, --help` | Show this help message and exit |
| `--data-root DATA_ROOT` | Path to directory containing slide subfolders |
| `--geneinfo GENEINFO` | Path to geneInfo file (tab/tsv/csv) |
| `--out-root OUT_ROOT` | Path to output directory |
| `--save-formats {binary,raw,cp10k_log1p} ...` | Output matrix formats |
| `--no-aac-features` | Do not create AAC/alt-read feature columns in spots table (alt_count_sum and derived ratio) |
| `--table-format {csv,tsv}` | Output table format for genes/spots |
| `-v, --verbose` | Increase verbosity (`-v`, `-vv`) |

### Example
 
```bash
python code/utils/preprocessing.py --data-root test_data --geneinfo geneInfo.tab --out-root /preprocessing/sample --save-formats binary --verbose
```
 
---
## 2. Train Model
 
Use `train.py` to train the model.
 
```
usage: train.py [-h] --out-root OUT_ROOT --slides SLIDES [SLIDES ...] [--val-slides [VAL_SLIDES ...]]
                [--test-slides [TEST_SLIDES ...]] [--hvg-json HVG_JSON] [--load-checkpoint LOAD_CHECKPOINT]
                [--matrix-type {binary,raw,cp10k_log1p}] [--epochs EPOCHS] [--lr LR]
                [--weight-decay WEIGHT_DECAY] [--batch-size BATCH_SIZE] [--seed SEED] [--exp-dir EXP_DIR]
                [--no-amp] [--compile] [--mlp-dims MLP_DIMS] [--cls-hidden-dim CLS_HIDDEN_DIM]
                [--prop-hidden-dim PROP_HIDDEN_DIM] [--input-mode {binary,expression}]
                [--use-class-head | --no-class-head] [--presence-dropout PRESENCE_DROPOUT]
                [--classifier-dropout CLASSIFIER_DROPOUT] [--layer-norm-mode {none,all,first,second}]
                [--class-lo-thr CLASS_LO_THR] [--class-hi-thr CLASS_HI_THR]
                [--class-label-smoothing CLASS_LABEL_SMOOTHING] [--class-weight-maternal CLASS_WEIGHT_MATERNAL]
                [--class-weight-fetal CLASS_WEIGHT_FETAL] [--class-weight-mix CLASS_WEIGHT_MIX]
                [--loss-type {ce,focal}] [--focal-gamma FOCAL_GAMMA] [--cls-loss-weight CLS_LOSS_WEIGHT]
                [--prop-loss-weight PROP_LOSS_WEIGHT] [--prop-consistency-weight PROP_CONSISTENCY_WEIGHT]
                [--aac-purity-weight AAC_PURITY_WEIGHT] [--prop-loss-type {huber,mse}]
                [--aac-column AAC_COLUMN] [--early-stop-patience EARLY_STOP_PATIENCE]
                [--monitor {cls_acc,prop_mae,prop_corr}] [--min-epochs MIN_EPOCHS] [--min-delta MIN_DELTA]
                [--eval-every EVAL_EVERY]
```
### Options
 
| Option | Description |
|---|---|
| `-h, --help` | Show this help message and exit |
| `--out-root OUT_ROOT` | Preprocessed data root containing genes/spots and count matrices |
| `--slides SLIDES [...]` | Training slide IDs (space-separated) |
| `--val-slides [...]` | Validation slide IDs (optional) |
| `--test-slides [...]` | Test slide IDs (optional) |
| `--hvg-json HVG_JSON` | Optional JSON file listing genes to use (e.g., HVGs) |
| `--load-checkpoint LOAD_CHECKPOINT` | Checkpoint to initialize or resume training from |
| `--matrix-type {binary,raw,cp10k_log1p}` | Expression source matrix. For input-mode=binary, use matrix-type=binary |
| `--epochs EPOCHS` | Maximum number of training epochs |
| `--lr LR` | Initial learning rate for AdamW |
| `--weight-decay WEIGHT_DECAY` | Weight decay (L2 regularization) for optimizer |
| `--batch-size BATCH_SIZE` | Spots per batch. Lower this if you use many genes |
| `--seed SEED` | Random seed for reproducibility |
| `--exp-dir EXP_DIR` | Directory to save checkpoints, logs, and predictions |
| `--no-amp` | Disable mixed precision |
| `--compile` | Enable torch.compile for faster throughput (requires PyTorch >= 2.0) |
| `--mlp-dims MLP_DIMS` | MLP encoder hidden dimensions, e.g. 1024,256,64 (default: 1024,256,64) |
| `--cls-hidden-dim CLS_HIDDEN_DIM` | Hidden dimension of the classification head MLP |
| `--prop-hidden-dim PROP_HIDDEN_DIM` | Hidden dimension of the proportion regression head MLP |
| `--input-mode {binary,expression}` | Model input interpretation: binary presence or expression values |
| `--use-class-head` | Enable maternal/fetal/mix classification head |
| `--no-class-head` | Disable classification head and train proportion-only model |
| `--presence-dropout PRESENCE_DROPOUT` | Randomly hide observed genes during training for robustness |
| `--classifier-dropout CLASSIFIER_DROPOUT` | Dropout rate used in classifier/regression heads |
| `--layer-norm-mode {none,all,first,second}` | LayerNorm placement in encoder hidden blocks |
| `--class-lo-thr CLASS_LO_THR` | Lower threshold for assigning maternal class from p_true |
| `--class-hi-thr CLASS_HI_THR` | Upper threshold for assigning fetal class from p_true |
| `--class-label-smoothing CLASS_LABEL_SMOOTHING` | Label smoothing factor for cross-entropy |
| `--class-weight-maternal CLASS_WEIGHT_MATERNAL` | Class weight for maternal samples in classification loss |
| `--class-weight-fetal CLASS_WEIGHT_FETAL` | Class weight for fetal samples in classification loss |
| `--class-weight-mix CLASS_WEIGHT_MIX` | Class weight for mix samples in classification loss |
| `--loss-type {ce,focal}` | Classification loss: standard cross-entropy (ce) or focal loss (focal) |
| `--focal-gamma FOCAL_GAMMA` | Focusing parameter for focal loss |
| `--cls-loss-weight CLS_LOSS_WEIGHT` | Weight of classification loss term |
| `--prop-loss-weight PROP_LOSS_WEIGHT` | Weight of proportion regression loss term |
| `--prop-consistency-weight PROP_CONSISTENCY_WEIGHT` | Weight of class-proportion consistency regularization |
| `--aac-purity-weight AAC_PURITY_WEIGHT` | Weight of AAC purity regularization term |
| `--prop-loss-type {huber,mse}` | Regression loss for fetal proportion prediction |
| `--aac-column AAC_COLUMN` | AAC source column. Use auto to search alt_count_sum/aac/alt_reads |
| `--early-stop-patience EARLY_STOP_PATIENCE` | Stop training after this many non-improving evals |
| `--monitor {cls_acc,prop_mae,prop_corr}` | Validation metric used for best-checkpoint selection and early stopping |
| `--min-epochs MIN_EPOCHS` | Minimum epochs to run before early stopping can trigger |
| `--min-delta MIN_DELTA` | Minimum improvement required to reset early-stopping patience |
| `--eval-every EVAL_EVERY` | Run validation every N epochs |
 
### Examples
 
```bash
python code/train.py --out-root /preprocessing/sample --slides train1 --valid-slides train2 --test-slides test --exp-dir /train --epochs 100 --batch-size 64 --mlp-dims 2048,512,128 --prop-hidden-dim 64 --cls-hidden-dim 64 --lr 1.52e-4 --weight-decay 1e-4 --class-lo-thr 0.1 --class-hi-thr 0.9 --no-amp --seed 42 --class-weight-maternal 1.0 --class-weight-fetal 1.0 --class-weight-mix 1.6 --class-label-smoothing 0.00 --early-stop-patience 25 --presence-dropout 0.15 --classifier-dropout 0.2 --prop-loss-weight 1.7 --prop-consistency-weight 0.2 --aac-purity-weight 0.05 --prop-loss-type mse --monitor prop_corr --layer-norm-mode second --loss-type focal --focal-gamma 3.5 --cls-loss-weight 1.82
```
 
```bash
python code/train.py --out-root /preprocessing/sample --slides train1 train2 --valid-slides test --exp-dir /train --epochs 100 --batch-size 64 --mlp-dims 2048,512,128 --prop-hidden-dim 64 --cls-hidden-dim 64 --lr 1.52e-4 --weight-decay 1e-4 --class-lo-thr 0.1 --class-hi-thr 0.9 --no-amp --seed 42 --class-weight-maternal 1.0 --class-weight-fetal 1.0 --class-weight-mix 1.6 --class-label-smoothing 0.00 --early-stop-patience 25 --presence-dropout 0.15 --classifier-dropout 0.2 --prop-loss-weight 1.7 --prop-consistency-weight 0.2 --aac-purity-weight 0.05 --prop-loss-type mse --monitor prop_corr --layer-norm-mode second --loss-type focal --focal-gamma 3.5 --cls-loss-weight 1.82
```
 
---

## 3. Load Model and Weights (Inference)
 
Use `inference.py` for inference.
 
```
usage: inference.py [-h] --checkpoint CHECKPOINT --preprocess-dir PREPROCESS_DIR
                     (--slide-id SLIDE_ID | --slides SLIDES [SLIDES ...] | --domain {cancer,covid,human,sim})
                     [--output-dir OUTPUT_DIR] [--per-slide-subdir] [--matrix-type {binary,cp10k_log1p}]
                     [--svd-dir SVD_DIR] [--aac-column AAC_COLUMN] [--batch-size BATCH_SIZE]
```

### Options
 
| Option | Description |
|---|---|
| `-h, --help` | Show this help message and exit |
| `--checkpoint CHECKPOINT` | Path to DeepSetMA/DeepFM/GatedAttn checkpoint |
| `--preprocess-dir PREPROCESS_DIR` | Directory containing preprocessed data |
| `--slide-id SLIDE_ID` | Single slide ID to run inference on |
| `--slides SLIDES [...]` | One or more slide IDs to run inference on |
| `--domain {cancer,covid,human,sim}` | Use built-in slide list for a domain |
| `--output-dir OUTPUT_DIR` | Output directory |
| `--per-slide-subdir` | Save outputs under `<output-dir>/<slide_id>` |
| `--matrix-type {binary,cp10k_log1p}` | Preprocessed matrix type to use for inference. Use binary for base/class/noaac and cp10k_log1p for expression |
| `--svd-dir SVD_DIR` | Optional SVD feature directory with `counts_svd_<slide>.npz` |
| `--aac-column AAC_COLUMN` | AAC source column name or auto |
| `--batch-size BATCH_SIZE` | Inference batch size (0 = full slide) |
 
### Examples
 
```bash
python code/utils/inference.py --checkpoint /train/best.ckpt --preprocess-dir /preprocessing/sample --slide-id test --output-dir /inf/sample --matrix-type binary
```
 
```bash
python code/utils/inference.py --checkpoint /golden.ckpt --preprocess-dir /preprocessing/sample --slides test train1 train2 --output-dir /inf/sample --matrix-type binary
```
 
---

## 4. Visualization
 
Use `visual.py` for visualization.
 
```
usage: visual.py [-h] --inference-dir INFERENCE_DIR
                  (--slide-id SLIDE_ID | --slides SLIDES [SLIDES ...] | --domain {cancer,covid,human,sim})
                  [--inference-per-slide-subdir] [--spots-root SPOTS_ROOT] [--in-tissue-only]
                  [--point-size POINT_SIZE]
```
 
*Saves plots to `<slide_dir>/plots/`*
 
### Options
 
| Option | Description |
|---|---|
| `-h, --help` | Show this help message and exit |
| `--inference-dir INFERENCE_DIR` | Directory with prediction CSVs, or base dir containing per-slide subdirs |
| `--slide-id SLIDE_ID` | Single slide ID to visualize |
| `--slides SLIDES [...]` | One or more slide IDs to visualize |
| `--domain {cancer,covid,human,sim}` | Use built-in slide list for a domain |
| `--inference-per-slide-subdir` | Read predictions from `<inference-dir>/<slide_id>/pred_<slide>.csv` |
| `--spots-root SPOTS_ROOT` | Root containing spots.* (needed if x/y not in prediction files) |
| `--in-tissue-only` | Keep only in_tissue==1 when spots table is available |
| `--point-size POINT_SIZE` | Point size |
 
### Examples
 
```bash
python code/utils/visual.py --inference-dir /inf/sample --slide-id test --spots-root /preprocessing/sample --in-tissue-only --point-size 9
```
 
```bash
python code/utils/visual.py --inference-dir /inf/sample2 --slides test train1 train2 --inference-per-slide-subdir --spots-root /preprocessing/sample --in-tissue-only --point-size 9
```
