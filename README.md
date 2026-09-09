This repository is the implementation of our Paper under review.


# ParaLoRA and ParaDG for Imbalanced Paratope Prediction


# Abstract
Accurate prediction of antibody paratopes is critical for elucidating immune recognition mechanisms and accelerating the discovery of therapeutic antibodies. Current deep learning approaches face three major limitations: severe class imbalance in benchmark datasets, prohibitive computational costs associated with full fine-tuning of large protein language models (PLMs), and inadequate integration of three-dimensional (3D) structural information. We present a unified computational framework that jointly exploits sequence and structural features for paratope prediction. For sequence-based modeling, we develop ParaLoRA, a lightweight parameter-efficient fine-tuning (PEFT) method that incorporates Low-Rank Adaptation (LoRA) into PLMs. ParaLoRA enables task adaptation for paratope prediction with only a small number of additional trainable parameters. It further mitigates class imbalance via targeted sampling and cost-sensitive weighting. For structure-based modeling, we design ParaDG, a dual-view graph learning model. ParaDG constructs two complementary antibody graphs to capture global folding topology and local surface microenvironments. A novel graph oversampling module balances the feature distribution while preserving native spatial connectivity. The augmented graph is processed through a dual-view graph neural network for multi-scale feature fusion. Comprehensive benchmarking demonstrates that ParaLoRA and ParaDG outperform state-of-the-art sequence-based and structure-based predictors, respectively. This work delivers a robust and scalable framework for paratope prediction and supports advances in antibody engineering research. 

Two complementary models for antibody paratope prediction:

- **ParaLoRA** — parameter-efficient fine-tuning of the frozen [ProtT5](https://github.com/agemagician/ProtTrans) encoder with LoRA on the attention matrices (Wq/Wk/Wv/Wo), reducing trainable parameters from 1.2 B to ~2.5 M (0.2 %) while keeping sequence-level paratope accuracy competitive with state-of-the-art sequence-based methods.
- **ParaDG** — a dual-view graph neural network over the antibody residue graph (global view + surface view with), trained with a structure-preserving graph oversampling module that balances the ~5 % positive residues without breaking spatial topology.

| Branch | Package | Training entrypoint |
| ------ | ------- | ------------------- |
| Sequence (ParaLoRA) | `paralora` | `python -m scripts._train_paralora_ft --train-data data/paralora/train.csv --valid-data data/paralora/val.csv ...` |
| Structure (ParaDG)  | `paradg`   | `python -m paradg.train --config configs/paradg.json --data-dir <your-dataset-dir> ...` |

## Repository layout
```
├── README.md / README_CN.md  -- this file / Chinese version
├── LICENSE                   -- MIT
├── requirements.txt          -- pip dependencies
├── environment.yml           -- conda equivalent
├── configs/                  -- JSON hyperparameters + experiment design notes
│   ├── paralora.json             -- sequence branch
│   ├── paradg{,_v2,_v3}.json     -- structure branch (v3 = released recipe)
│   ├── paralora_finetune_design.md
│   └── paralora_ablation_design.md
├── paralora/                 -- sequence branch library
│   ├── lora.py              -- LoRAConfig / LoRALinear / modify_with_lora
│   ├── model.py             -- T5EncoderForTokenClassification + LoRA wiring
│   ├── data.py              -- CSV / paraperd / pkl split loaders
│   ├── trainer.py           -- train_per_residue (HF Trainer + DeepSpeed)
│   └── evaluate.py          -- metrics and embedding extraction
├── paradg/                   -- structure branch library
│   ├── models.py            -- ParaDG dual-view model + WeightedBCELoss
│   ├── data.py              -- dual-view graph construction (16 Å + rASA)
│   ├── oversampling.py      -- Algorithm 1: structure-preserving oversampling
│   ├── evaluate.py          -- metrics, CI aggregation, paired t-test
│   └── train.py             -- multi-seed training CLI
├── data_prep/               -- end-to-end dataset construction pipeline
│   ├── fetch_pdb_files.py           -- download/collect PECAN PDB files
│   ├── build_structure_dataset.py    -- Cα coords + DSSP rASA + labels
│   ├── embed_with_prot_t5.py         -- generic ProtT5 residue embeddings
│   ├── prepare_sequence_splits.py    -- Parapred sequence CSV splits
│   ├── build_structure_from_pkls.py -- structure pkl for an existing feature set
│   └── merge_features_struct.py     -- merge feature + structure pickles
├── analysis/                -- sanity checks and paper statistics
│   ├── graph_connectivity.py
│   ├── cdr_mask_statistics.py
│   ├── profile_efficiency.py
│   └── significance_test.py
├── scripts/                 -- training / embedding / ablation entrypoints
│   ├── _train_paralora_custom.py    -- shared training helpers
│   ├── _train_paralora_ft.py       -- ParaLoRA fine-tuning (main)
│   ├── _cv_paralora.py             -- 10-fold cross-validation
│   ├── _ablation_cv_paralora.py    -- placement/rank/alpha ablations
│   ├── _gen_paralora_embeddings.py -- dump fine-tuned residue embeddings
│   ├── _sanity_{paralora,paradg}.py-- smoke tests
│   └── run_all.py                  -- chained data-prep convenience CLI
├── notebooks/               -- Jupyter entry points (outputs cleared)
└── data/                    -- small text datasets (committed)
    ├── paralora/            -- Parapred sequence splits (552 complexes)
    ├── paralora_str/        -- sequence splits with structure columns
    └── splits/              -- official PECAN split lists (train/val/test)
```
## Installation

```bash
# pip
pip install -r requirements.txt

# or conda
conda env create -f environment.yml
conda activate paralora
```

Tested with Python 3.9, PyTorch 2.1–2.3 (CUDA 11.8/12.1), a single consumer-grade GPU (≥ 16 GB) is enough for ParaLoRA fine-tuning; ParaDG trains in minutes on one GPU.

## Data

Two datasets are used by the paper and both can be rebuilt from this repo:

1. **Parapred sequence splits** — already included under `data/paralora/`; columns`sequence,label,mask` with per-residue binary paratope labels). No download needed to train the sequence branch.
2. **PECAN structure dataset** — rebuild it from the official split lists in `data/splits/pecan-paratope-{train,val,test}.txt`:

   ```bash
   python -m scripts.run_all fetch-pdb \
       --split-list-dir data/splits \
       --src-complex /path/to/AbAg-pdb-chain \
       --src-ab /path/to/Ab-pdb --src-ag /path/to/Ag-pdb \
       --out-dir data/pdb

   python -m scripts.run_all structure-dataset \
       --data-dir data/pdb --out-dir data/pecan --rasa-threshold 0.25

   python -m scripts.run_all embed \
       --input-dir data/pecan --output-dir data/pecan
   ```

   `build_structure_dataset.py` requires the DSSP executable
   (`conda install -c conda-forge dssp`, version 4.x) for rASA computation.

## Usage

### 1. Fine-tune ParaLoRA (sequence branch)

```bash
python -m scripts._train_paralora_ft \
    --base-config configs/paralora.json \
    --train-data data/paralora/train.csv \
    --valid-data data/paralora/val.csv \
    --test-data  data/paralora/test.csv \
    --output-dir runs/paralora \
    --lora-layers q|k|v|o --lora-rank 4 \
    --class-weight balanced --eval-scope all
```

### 2. Export fine-tuned embeddings for ParaDG

```bash
python -m scripts._gen_paralora_embeddings \
    --config configs/paralora.json \
    --src-dir /path/to/structure-pickles \
    --out-dir /path/to/paralora-features
```

### 3. Train ParaDG (structure branch)

```bash
python -m paradg.train \
    --config configs/paradg_v3.json \
    --data-dir /path/to/paralora-features \
    --save-dir results/paradg \
    --seeds 0 1 2 3 4 \
    --graph-source coords --distance-threshold 16.0 \
    --surface-mode feature \
    --synthetic-edges incoming --k-neighbors 7 \
    --target-pos-ratio 0.25 --lambda-max 0.02
```

## License

Released under the MIT License (see `LICENSE`).
