# Synthetic Vibration-Signal Generation for Bearing Fault Diagnosis under Label Scarcity
### Reverse-Engineered Project Report & Presentation Pack

> **Provenance legend used throughout this document**
> **[V] Verified** — read directly from code/data in the repository.
> **[I] Inferred** — a defensible conclusion from repository evidence, not stated explicitly.
> **[U] Unknown / Missing** — not determinable from the repo; flagged as a gap.

---

## 0. Executive Summary (1 page — for the supervisor)

**What the project is. [V/I]**
The repository builds and evaluates **synthetic time-series generators** for the **CWRU rolling-bearing vibration dataset** (12 fault classes, sampled at 12 kHz). The central research question is: *can synthetic vibration signals replace scarce real labelled data for fault classification without losing accuracy?* The project answers this by (a) generating synthetic signals with **four independent strategies**, (b) checking spectral fidelity in the **FFT domain**, and (c) measuring the **downstream classification accuracy** when real training data is progressively replaced by synthetic data.

**The four generation strategies explored. [V]**
1. **GPT-4.1 prompting** (`prompting_gpt4/`) — zero-shot, prompt a frontier LLM to emit a sinusoid-like signal.
2. **GPT-2 LoRA fine-tuning** (`finetuning/`) — fine-tune GPT-2-124M on numeric values serialized as text.
3. **Local transformer "LLM"** (`local-llm/`) — a 17.4 M-parameter, 8-layer transformer trained to reconstruct 256-sample windows while matching time- and frequency-domain statistics.
4. **WGAN-GP family** (`WGAN-GP/`) — class-conditional Wasserstein GANs; ~11 architecture iterations ending in a **TCN + self-attention** generator with an **auxiliary classifier (ACGAN-style) critic**. This is the **primary, most-developed line of work**.

**The evaluation harness. [V]**
A **Temporal Convolutional Network (BigTCN)** classifier (`classifier/`, `full_pipeline.py`) is the yard-stick. Experiments sweep the **real-vs-synthetic training mix** (0–90 % synthetic) and measure accuracy on a **held-out 100 %-real test set**, per-class and overall, averaged over 10 repeats.

**Headline result. [V]** (`total_based_gen_mix_train90_summary.csv`)
Replacing real training windows with WGAN-GP self-attention synthetic windows degrades overall test accuracy **gracefully**: **98.5 %** (0 % synthetic) → **98.6 %** (30 %) → **95.2 %** (50 %) → **84.7 %** (90 %). Up to ~30–40 % synthetic substitution is **essentially free**; quality collapses only under heavy substitution, and unevenly across classes (7OR and 14BA degrade first; N, 7IR, 7BA stay at 100 %).

**Maturity. [I]** The WGAN-GP→TCN classifier loop is the mature, end-to-end pipeline (`full_pipeline.py`). The GPT-4/GPT-2/local-LLM lines are exploratory generators evaluated mostly by qualitative FFT/time-domain plots, not yet plugged into the quantitative classifier sweep.

**Biggest technical risk. [I]** Inconsistent preprocessing across modules (min-max [-1,1] in the GAN pipeline vs. per-window z-score in the classifier harness) and **no held-out validation of GAN training stability metrics (FID is plotted but not tabulated)** make cross-experiment comparison fragile. The "synthetic data works" claim currently rests on **one** generator family and **one** committed results table.

**Highest-impact next step. [I]** Unify preprocessing + a single config, then run the full 4-generator × mix-fraction sweep through the *same* classifier to produce one apples-to-apples results table (generator quality ranked by downstream accuracy, not eyeballed spectra).

---

## 1. Primary Research Question & Importance

**Problem. [V/I]** Rolling-element bearing failures are a dominant cause of rotating-machinery downtime. Data-driven fault diagnosis needs large labelled vibration datasets, but **labelled fault data is scarce** (faults are rare, seeding real faults is expensive/destructive). The project asks:

> *Can we synthesize realistic, class-conditioned bearing-vibration signals good enough that a classifier trained partly (or mostly) on synthetic data still generalizes to real signals?*

**Why it matters. [I]**
- **Data augmentation under label scarcity** — directly relevant to predictive maintenance / Industry-4.0.
- **Class imbalance** — fault classes are under-represented vs. "Normal"; conditional generation can rebalance.
- **Benchmark for generative models on 1-D physical signals** — vibration signals have sharp, physically-meaningful spectral peaks (bearing characteristic frequencies), so **frequency-domain fidelity** is a stringent, falsifiable quality test that image-GAN intuition misses.

---

## 2. Dataset

**Source. [V/I]** CWRU (Case Western Reserve University) bearing dataset — the de-facto standard benchmark for bearing fault diagnosis. Sampling frequency **fs = 12 000 Hz** (used everywhere: `compare_data.py`, `v6.md`).

**Structure. [V]**
- `CWRU_data/` — 12 raw single-column CSVs, each a long 1-D signal (~243 k samples per class, e.g. `N.csv` 243 938 rows, `7BA.csv` 243 538 rows).
- `imp4/` — the **same 12 classes pre-windowed** to `1824 × ~100` (1824 samples per window, ~100 windows per class file). [I] "imp4" = an earlier windowing export.
- `data.csv` (327 680 rows) / `train_data_70.csv` / `test_data_30.csv` — a single-signal 70/30 split used by the early local-LLM / FFT experiments. [I]

**12-class label taxonomy. [V]** (`full_pipeline.py`, `classifier/model_0.8train.py`)
`N` (normal) + fault **location** × **size**:
`7BA 7IR 7OR · 14BA 14IR 14OR · 21BA 21IR 21OR · BA28 IR28` — where `BA`=ball, `IR`=inner race, `OR`=outer race, and `7/14/21/28` = fault diameter in mils. Some classifier variants collapse this to **4 classes** (Normal / IR / BA / OR) for inter-load generalization studies.

**Window length. [V]** Standardized to **SEQ_LEN = 1824** for the GAN + classifier pipeline (`full_pipeline.py`, `model_0.8train.py`); the local-LLM uses **256** windows (power of two for FFT efficiency, per `v6.md`).

**Preprocessing — INCONSISTENT across modules. [V] (important caveat)**
- GAN pipeline (`full_pipeline.py`): min-max to **[-1, 1]** per file.
- Classifier harness (`model_0.8train.py`): **z-score** (per-signal for real, **per-column** for generated).
- → [I] A latent confound when comparing experiments.

**Splits. [V]** `StratifiedShuffleSplit`, train fraction 0.8–0.9, **test set always 100 % real**, 10 repeats with varying seeds.

---

## 3. Methodology — Overall Pipeline

```
                 ┌────────────────────────────────────────────────────────┐
                 │                  CWRU raw signals (12 kHz)               │
                 └────────────────────────────────────────────────────────┘
                                          │  window (1824 / 256), normalize
                                          ▼
        ┌──────────────────────────── GENERATION (4 parallel strategies) ───────────────────────────┐
        │                                                                                            │
        │  (A) GPT-4.1 prompt   (B) GPT-2 LoRA      (C) Local transformer     (D) WGAN-GP (primary)  │
        │      zero-shot            fine-tune            17.4M, recon+stats        TCN+SelfAttn G,    │
        │                                                                          ACGAN TCN critic   │
        └──────────────┬─────────────────┬─────────────────────┬──────────────────────┬─────────────┘
                       │                 │                     │                      │
                       ▼                 ▼                     ▼                      ▼
                 synthetic CSVs    synthetic CSVs       data_generated.csv    WGAN-GP/<variant>/<class>.csv
                       │                 │                     │                      │
                       └────────────┬────┴──────────┬──────────┴──────────────┬───────┘
                                    ▼               ▼                          ▼
                         (1) SPECTRAL FIDELITY CHECK            (2) DOWNSTREAM UTILITY CHECK
                         compare_data.py / fft_*.py             classifier/*.py, full_pipeline.py
                         FFT magnitude, Pearson, cosine,        BigTCN trained on real+synthetic mix,
                         KL, MMD per class (best-of-N col)      tested on 100% real → accuracy curves
                                    │                                          │
                                    └─────────────────► best generator ◄───────┘
```

**Stage explanations.**
1. **Windowing/normalize [V]** — segment long signals into fixed windows; normalize.
2. **Generation [V]** — four strategies emit synthetic windows; only WGAN-GP is class-conditional and produces one CSV per class (`1824 × N` columns, each column = one synthetic window).
3. **Spectral check [V]** (`compare_data.py`, `fft_generator_script.py`) — FFT each window; for each class pick the **best-matching synthetic column** by Pearson/cosine (max) and KL/MMD (min); plot real-vs-synthetic spectra and time series; save `fft_comparison_best_columns.png`, `timeseries_comparison_best_columns.png`.
4. **Utility check [V]** (`classifier/`, `full_pipeline.py`) — train BigTCN on a real/synthetic mix, evaluate on real test set; sweep mix fraction; save curves + summary CSV.

---

## 4. Model Architectures

### 4.1 WGAN-GP generator family (primary line) [V]
A clear **11-step architectural evolution** (dates from `ls -la`):

| # | File | Date | Generator | Critic | Conditional | Output len | Key change |
|---|------|------|-----------|--------|-------------|-----------|-----------|
| 1 | `model.py` | Aug-09 | Transformer (4 blk) | ViT-style patch critic | no | 300 | first version; FFT + feature-matching loss |
| 2 | `model_v0.py` | Aug-09 | Transformer + dilated conv | multi-stream (time+freq) critic | no | 128 | spectral PSD/phase loss |
| 3 | `model_v0.2.py` | Aug-15 | latent-upsample + transformer + conv | dual-stream time/STFT, 3 critics | no | 128 | MR-STFT loss, instance-noise anneal, patch critics |
| 4 | `model_cnn.py` | Aug-16 | dual-branch CNN (low/high-freq) | multi-scale CNN | no | 1824 | pure CNN, freq-aware branches |
| 5 | `model_tcnn_in_critic.py` | Aug-27 | ConvTranspose CNN | **TCN + aux classifier (ACGAN)** | **yes (4 cls)** | 1824 | class-conditional + classification loss |
| 6 | `model_full_tcn_class.py` | Aug-29 | upsample→TCN→downsample | TCN ACGAN | yes (10 cls) | 1824 | full-TCN generator |
| 7 | `model_full_tcn_class2.py` | Sep-03 | 4-stage explicit upsample + TCN | TCN ACGAN | yes (**12 cls**) | 1824 | +3 classes (BA28/IR28), explicit upsampling |
| 8 | `model_new.py` | Sep-19 | TCN + **pluggable attention** (self/MH/channel/temporal/mixed) | attention TCN | yes (12) | 1824 | attention ablation harness |
| 9 | `model_hybrid.py` | Oct-18 | TCN + enhanced attn (spectral-norm, pre-norm, SE, temp-scaling, layer-scale) | enhanced TCN | yes (12) | 1824 | stability hardening |
| 10 | `model_only_tcn_attention.py` | Jan-07 | 12 TCN + 8 transformer blocks, depthwise-sep conv, SE | TCN+transformer | yes (12) | 1824 | deep hybrid |
| 11 | **`model_self_attention.py`** | **Jan-29 (latest)** | **AttentionTCNBlock ×9 (self-attn per block) + global self-attn** | **TCN ACGAN dual-head** | yes (12) | 1824 | **current "production" generator** |

**Final/representative model (verified in `full_pipeline.py`, copied from `model_self_attention.py`):**
- **Generator `GeneratorTCNWithSelfAttention`**: label embedding → `ConvTranspose1d` init (length 114) → 4× upsample → **9 `AttentionTCNBlock`s** (dilations 2⁰…2⁸, each = 2×Conv1d + LayerNorm + residual + `SelfAttention`) → global `SelfAttention` → downsample to 1 channel → `tanh`. nz=100, channels=64.
- **Critic `DiscriminatorTCN`**: Conv1d → 9 `TCNBlock`s → `AdaptiveAvgPool1d` → **two heads**: `adv_output` (Wasserstein score) + `classifier` (12-way). → **ACGAN**.
- **Loss [V]**: `L_D = E[D(fake)] − E[D(real)] + λ_gp·GP + CE(class|real)`, `L_G = −E[D(fake)] + CE(class|fake)`, with **λ_gp = 10**, gradient penalty on interpolates. (Note: the *standalone* GAN scripts add explicit **FFT magnitude/phase/PSD losses**; `full_pipeline.py` keeps only WGAN-GP + classification for speed.)

### 4.2 Local transformer "LLM" generator [V] (`local-llm/model_llm_v6.py`, `v6.md`)
- **Input**: 256-sample window **+ 47 hand-engineered features** (12 time, 17 frequency, 6 advanced-stat, 3 spectrogram, 3 nonlinear/entropy, 6 sensor-specific).
- **Backbone**: 8-layer Transformer encoder, **d_model 512, 8 heads, d_ff 1024, dropout 0.05**, learnable positional embedding; signal-projection + feature-projection (features scaled ×0.01 to avoid dominating). **≈17.4 M params** (`best_model.pth` 4.2 MB state-dict; exported `multihead_model.onnx` 71 MB).
- **Output heads**: reconstructed signal with a **learnable residual blend** `y = σ(α)·x + (1−σ(α))·ŷ` (α init 0.8) + a **statistics head** predicting (mean, std, skew, kurtosis, range).
- **Loss [V]**: weighted sum of MSE (1.5), std (0.9), mean/range/percentile (0.3), skew/kurtosis (0.2), **frequency-domain (1.2)**, stats-head (0.4), Pearson (0.5), cosine (0.5); AdamW lr 5e-5, grad-clip 0.1.
- **Generation [V]**: **seed-based reconstruction** (not unconditional) — feed real windows as seeds, model transforms them; 200 variance-stratified seeds → 48 000 synthetic samples (`data_generated.csv`).

### 4.3 GPT-2 LoRA generator [V] (`finetuning/fine-tuning-gpt-2.py`)
- Base **GPT-2-124M**, **LoRA** (r=8, α=16, target `c_attn`/`c_proj`, dropout 0.05), fp16, lr 2e-4, effective batch 16, ~50 steps on first 1000 values serialized as `"Generate sensor data: v1, v2, …"`.
- Sampling: templated prompts, temperature 0.8, top-p 0.9 → parse numbers back → `gpt2-fine-tuned-sensor-data.csv` (5000×10). [I] Crude numeric-from-text decoding with Normal(0,1) fallback — weakest fidelity of the four.

### 4.4 GPT-4.1 prompting [V] (`prompting_gpt4/`, `approach.txt`)
- Single natural-language prompt asking GPT-4.1 to emit 1000 sinusoid-plus-noise samples → `gpt-4.1_data.csv`; evaluated only by `Time_domain_compare_with_gpt.png` / `FFT_compare_with-gpt.png`. [I] Baseline / sanity reference.

### 4.5 Classifier (evaluation backbone) [V] (`classifier/`, `full_pipeline.py`)
- **BigTCN**: 5 `TemporalBlock`s, channels [64,64,128,128,256], kernel 5, dilations 2⁰…2⁴, causal cropping, residual, `AdaptiveAvgPool1d` → `Linear` → 12 (or 4) classes. Input (1, 1824). Adam lr 1e-3, 25–30 epochs.

---

## 5. Training Strategy — Losses & Optimization [V]

| Component | Optimizer | LR | Key regularizers | Loss |
|-----------|-----------|----|------------------|------|
| WGAN-GP G/D | Adam β=(0.5,0.9) | 1e-4 | gradient penalty λ=10, BatchNorm, (spectral-norm in hybrid) | Wasserstein + GP + ACGAN cross-entropy (+FFT in standalone scripts) |
| Local LLM | AdamW | 5e-5 | grad-clip 0.1, dropout 0.05, StepLR | 11-term recon + statistical + frequency + correlation loss |
| GPT-2 | AdamW (HF Trainer) | 2e-4 | LoRA, fp16, warmup 10 | causal LM cross-entropy |
| BigTCN classifier | Adam | 1e-3 | — | cross-entropy |

**Self-supervised / contrastive components requested in the brief: [V] none present.** There is **no contrastive or SSL pretraining**. "Self-attention" here is architectural (intra-signal attention), and "supervision" in the local-LLM is reconstruction + statistical matching, not contrastive. (Flagging because the prompt asked specifically.)

---

## 6. Experimental Setup [V/U]

- **Hardware [I/U]**: CUDA-if-available (`DEVICE = cuda or cpu`); exact GPU **[U]** — not recorded. Vendored `bitsandbytes/` [I] implies 8-bit/quantized experimentation, likely to fit GPT-2/LLM work on a single consumer GPU.
- **Repro [V]**: SEED=42 everywhere; 10 repeats per configuration with seed+r.
- **Key hyperparameters**: SEQ_LEN 1824 (GAN/classifier) or 256 (LLM); batch 32–64; GAN 50 epochs, classifier 25–30 epochs.
- **Dependencies [V]**: PyTorch, scikit-learn, scipy, pandas, numpy, matplotlib, tqdm, HuggingFace `transformers`+`peft`, `bitsandbytes`. **No `requirements.txt`/Dockerfile [U]** — environment is implicit.

---

## 7. Experiments & Ablations

| Experiment | Change | Motivation | Result | Status |
|-----------|--------|-----------|--------|--------|
| Generator architecture sweep | 11 WGAN-GP variants (transformer→CNN→TCN→TCN+attention) | find best 1-D signal generator | Converged on TCN+self-attention ACGAN | **[V]** code; ranking **[I]** |
| Attention-type ablation | self / multi-head / channel / temporal / mixed (`model_new.py`) | which attention helps spectral fidelity | harness exists; chosen = self-attention | **[V]** harness, **[U]** numbers |
| **Total-based real→synthetic replacement** | synthetic = 0–90 % of *total* train pool, 12 classes, 10 repeats | does synthetic data preserve accuracy? | 98.5%→98.6%(30%)→95.2%(50%)→**84.7%(90%)** | **[V]** `..._train90_summary.csv` |
| Per-class real-data scaling | real fraction 10–90 % (`model_v1.py`) | accuracy vs. label budget | curve `tcn_5000_per_class_scaling.png` | **[V]** plot, **[U]** table |
| Train-frac × gen-frac grid | real% × gen% (`model_v1_with_generated_data.py`) | augmentation utility surface | curves `train{10..90}_gen_mix.png` | **[V]** 9 plots |
| Inter-load generalization | train on loads {7,14}, test held-out load (`model_v2.py`, 4-class) | domain shift across operating loads | confusion matrices + curves | **[V]** code, **[U]** committed numbers |
| End-to-end GAN+classifier loop | train GAN per fraction, fill remainder, classify (`full_pipeline.py`) | self-contained augmentation pipeline | `gan_augmentation_results.png` (on run) | **[V]** code; **[U]** committed result |
| Spectral fidelity (FFT) | Pearson/cosine/KL/MMD best-column per class | quantify generator quality | `fft_comparison_best_columns.png` | **[V]** plot; **[U]** numeric table |
| Local-LLM epoch sweep | v6 at Ep 5–80, dropout 0.45 | training-length effect | loss/FFT/time plots per epoch | **[V]** plots |
| FID tracking (LLM & GAN) | FID over training | generator convergence | `fid-history--*.png` | **[V]** plot; **[U]** final value |

---

## 8. Results

**Quantitative (only fully-committed numeric result) [V]** — `total_based_gen_mix_train90_summary.csv`, BigTCN, 12-class, 10 repeats, test = 100 % real:

| Synthetic % of train | Overall acc | Std | Weakest classes |
|---|---|---|---|
| 0 % | **0.985** | 0.017 | 14BA 0.83 |
| 10 % | 0.984 | 0.014 | 14BA 0.80 |
| 20 % | 0.980 | 0.015 | 14BA 0.67 |
| 30 % | **0.986** | 0.013 | 14BA 0.87 |
| 40 % | 0.969 | 0.018 | 14OR 0.85 |
| 50 % | 0.952 | 0.034 | 14BA 0.77 |
| 60 % | 0.948 | 0.051 | 14BA 0.63 |
| 70 % | 0.932 | 0.050 | 14BA 0.67 |
| 80 % | 0.935 | 0.028 | 14BA 0.77 |
| 90 % | **0.847** | 0.056 | 7OR 0.57, 14BA 0.57 |

**Observations [V/I]:**
- **N, 7IR, 7BA stay at 1.00 across all mixes** → the GAN reproduces these classes near-perfectly. **[V]**
- **14BA (ball fault, 14-mil) is the consistently weakest** and the first to collapse → [I] the generator under-models this class's spectrum; a targeted quality problem, not a global one.
- The curve is **flat to ~40 % then bends** → [I] synthetic data is a near-free substitute up to ~⅓–½ of the training set; useful operating point for augmentation.
- Rising std at high synthetic% → [I] synthetic-heavy training is less stable.

**Qualitative [V]:** FFT-magnitude and time-domain overlays (`fft_comparison_best_columns.png`, `timeseries_comparison_best_columns.png`, `local-llm/FFT_compare_*`, `prompting_gpt4/FFT_compare_*`) show real-vs-synthetic spectra per class with annotated Pearson correlation per best column.

**Missing / not-yet-collected [U]:** F1 / precision / recall / mAP / IoU (not applicable or not computed); a numeric FID table; a committed results CSV for `full_pipeline.py` and `model_v2.py`; a single table ranking the four generators by downstream accuracy.

---

## 9. Current Progress

**Completed [V]**
- CWRU ingestion, windowing, 12-class labelling, stratified splits.
- Four generation strategies implemented and run; synthetic CSVs produced.
- WGAN-GP architecture matured through 11 iterations to TCN+self-attention ACGAN.
- FFT-domain spectral comparison tooling.
- BigTCN classifier + real/synthetic mix-sweep harness; **one full quantitative result table**.

**Partially completed [V/I]**
- Attention-type and inter-load studies (code present, numbers not committed).
- Local-LLM v7/v7.1 (CNN-transformer hybrid) — architecture txts exist; not clearly integrated into the classifier sweep.
- End-to-end `full_pipeline.py` (runs, but no committed output).

**Not started / missing [U]**
- Unified config/preprocessing; environment pinning (requirements/Docker).
- Apples-to-apples 4-generator benchmark via the same classifier.
- Statistical significance / confidence intervals beyond raw std; FID numeric reporting.
- README / reproducibility docs (README is empty).

---

## 10. Challenges

**Technical [V/I]**
- **GAN training stability** — the whole hybrid line (spectral-norm, pre-norm, SE, temperature-scaling, gradient penalty, instance-noise annealing) is evidence of fighting instability. **[V]**
- **Preprocessing inconsistency** (min-max vs z-score; per-signal vs per-column) across modules → confounds comparison. **[V]**
- **Class-specific generation failure** (14BA) → uneven spectral fidelity. **[V from results]**
- **Numeric-from-text decoding** in the GPT-2 path is brittle (regex parse + Gaussian fallback). **[V]**
- **No environment lock / large binaries committed** (`best_model.pth`, `.onnx`, vendored `bitsandbytes`, many CSVs) → repo hygiene + reproducibility risk. **[V]**

**Research [I]**
- **Domain adaptation across loads/RPM** (the `model_v2` study targets this directly).
- **Generalization vs memorization** — does the GAN synthesize novel windows or copy seeds? (the local-LLM is explicitly *seed-reconstruction*, so this risk is real). **[V for LLM]**
- **Label scarcity** is the framing motivation but real fault data is still required to *train* the generators.

---

## 11. Future Work (prioritized) [I]

**High**
1. Unify preprocessing + one config; re-run the headline sweep to remove the normalization confound.
2. Benchmark all four generators through the *same* BigTCN → single downstream-accuracy ranking table.
3. Diagnose & fix 14BA / 7OR generation (per-class spectral loss weighting, or class-balanced critic).

**Medium**
4. Commit numeric FID + spectral-metric tables (not just PNGs); add F1/precision/recall + confusion matrices to the main result.
5. Integrate local-LLM v7.1 hybrid into the quantitative loop; compare to WGAN-GP.
6. Test inter-load / inter-RPM generalization at scale (extend `model_v2`).

**Low**
7. Repo hygiene: `requirements.txt`/Dockerfile, move large binaries to LFS/release artifacts, fill README.
8. Hyperparameter search for the classifier; longer GAN schedules; export/ONNX inference path.

---

## 12. Research & Engineering Contributions [I, grounded in V]

- **Engineering:** a reusable **generate → spectral-check → downstream-utility** evaluation harness for 1-D signal generators, with a clean real/synthetic mix-fraction protocol and 10-repeat averaging.
- **Architectural:** a documented **11-step evolution** converging on a **class-conditional TCN + self-attention WGAN-GP (ACGAN critic)** for vibration signals, plus a pluggable attention-ablation framework.
- **Empirical:** evidence that **synthetic vibration data can substitute ~30–40 % of real training data at no accuracy cost**, with a characterized graceful-degradation curve and per-class failure modes.
- **Methodological:** insistence on **frequency-domain fidelity** (FFT magnitude/phase/PSD losses + FFT-based evaluation) as the right quality bar for physical signals — and a custom **feature-aware, statistics-matching transformer** generator.

---

## 13. Contribution Map (file → role) [V]

```
DATA
  CWRU_data/*.csv ............ raw 12-class CWRU signals (12 kHz)
  imp4/*.csv ................. pre-windowed (1824×~100) per class
  data.csv / train_data_70 / test_data_30 .. single-signal split (FFT/LLM expts)

GENERATION
  prompting_gpt4/ ............ GPT-4.1 zero-shot synthetic + compare PNGs
  finetuning/fine-tuning-gpt-2.py .. GPT-2 LoRA fine-tune → gpt2-...-data.csv
  local-llm/model_llm_v6.py .. 17.4M transformer recon generator → data_generated.csv
  local-llm/best_model.pth, multihead_model.onnx .. trained LLM weights
  WGAN-GP/model_self_attention.py .. CURRENT generator (TCN+self-attn ACGAN)
  WGAN-GP/model_*.py ......... 10 prior architecture iterations
  WGAN-GP/self attention/*.csv .. committed synthetic windows (per class, 1824×N)
  WGAN-GP/hybrid/*.csv ....... hybrid-model synthetic windows

EVALUATION — spectral
  compare_data.py ............ per-class FFT Pearson/cosine/KL/MMD best-column + plots
  fft_generator_script.py .... single-signal FFT/time plot

EVALUATION — downstream
  classifier/model_v1.py ..... BigTCN, real-only data-scaling (12-class)
  classifier/model_v2.py ..... BigTCN, inter-load generalization (4-class) + confusion
  classifier/model_v1_with_generated_data.py .. real×gen mix grid → train{10..90}_gen_mix.png
  classifier/model_0.8train.py .. total-based replacement sweep → *_train90_summary.csv
  full_pipeline.py ........... END-TO-END: train GAN per fraction → fill → classify

DOCS
  approach.txt ............... the 4-strategy plan
  v6.md ...................... full math of the local-LLM transformer
  README.md .................. EMPTY (gap)
```

## 14. Repository Dependency Graph [V/I]

```
CWRU_data ──► (windowing) ──► full_pipeline.py ──► trains WGAN-GP(self-attn) ──► synthetic ──┐
     │                                                                                       ├─► BigTCN ─► accuracy
     ├──► WGAN-GP/model_*.py ──► WGAN-GP/<variant>/*.csv ──────────────────────────┐         │
     │                                                                             ├─► classifier/model_0.8train.py ─► summary.csv
     │                                                                             │   classifier/model_v1_with_generated_data.py ─► gen_mix PNGs
     ├──► local-llm/model_llm_v6.py ──► data_generated.csv ──────────────┐         │
     ├──► finetuning/gpt-2 ──► gpt2-...-data.csv ───────────────────────┐ │        │
     ├──► prompting_gpt4 ──► gpt-4.1_data.csv ─────────────────────────┐│ │        │
     └──► compare_data.py ◄── (real + any synthetic dir) ─────────────► FFT metrics + PNGs
```
*Note: only the WGAN-GP → classifier path is wired end-to-end with committed numeric results; the GPT-4/GPT-2/local-LLM outputs currently feed only the spectral (FFT) comparison.*

---

## 15. Slide-by-Slide Presentation (18 slides)

> Each slide: **Objective · Bullets · Diagram · Speaker notes (~30–60 s)**

**Slide 1 — Title**
- *Objective:* Frame the talk.
- Bullets: "Synthetic Vibration-Signal Generation for Bearing Fault Diagnosis under Label Scarcity"; Author: Atharv Gaur; Supervisor: [name]; CWRU dataset, 12 kHz, 12 fault classes.
- *Diagram:* hero image — real vs. synthetic FFT overlay (`fft_comparison_best_columns.png`).
- *Speaker notes:* "Bearing faults cause most rotating-machine downtime, but labelled fault data is scarce. I ask whether we can generate synthetic vibration signals good enough to train fault classifiers, and I test four generation strategies against one downstream classifier."

**Slide 2 — Motivation**
- *Objective:* Why the problem matters.
- Bullets: faults rare & costly to seed; classifiers need lots of labels; class imbalance; predictive-maintenance impact.
- *Diagram:* simple "label scarcity" funnel.
- *Notes:* "Real fault data is expensive and imbalanced. If synthetic data can stand in, we cut labelling cost and rebalance rare classes."

**Slide 3 — Background**
- *Objective:* Domain primer.
- Bullets: CWRU benchmark; bearing characteristic frequencies → sharp FFT peaks; fault = location (IR/BA/OR) × size (7/14/21/28 mil); 12 kHz sampling.
- *Diagram:* annotated FFT of one class showing fault peak.
- *Notes:* "Vibration faults show up as specific spectral peaks, so frequency-domain fidelity is a strict, physical correctness test — harder than it looks."

**Slide 4 — Research Gap**
- *Objective:* What's missing in prior work.
- Bullets: most GAN work judged by eye / time-domain; little on *downstream-utility* of synthetic vibration data; no single comparison of LLM-prompting vs fine-tuning vs custom-transformer vs GAN.
- *Diagram:* 2×2 "evaluated by spectra vs by accuracy" / "one generator vs many".
- *Notes:* "I close two gaps: comparing four very different generators head-to-head, and judging them by classifier accuracy on real data, not just visual spectra."

**Slide 5 — Objective**
- *Objective:* State goals.
- Bullets: (1) generate class-conditional CWRU signals; (2) verify spectral fidelity; (3) measure accuracy as real data is replaced by synthetic; (4) find the best generator.
- *Diagram:* the master pipeline (Section 3).
- *Notes:* "Concretely: generate, FFT-check, then stress-test by swapping real for synthetic and watching classifier accuracy."

**Slide 6 — Repository Overview**
- *Objective:* Orient the audience.
- Bullets: `WGAN-GP/` (primary), `local-llm/`, `finetuning/`, `prompting_gpt4/`, `classifier/`, `CWRU_data/`, `full_pipeline.py`.
- *Diagram:* contribution map (Section 13).
- *Notes:* "Four generator folders, one classifier folder, one end-to-end pipeline, and the FFT comparison tooling tie it together."

**Slide 7 — Dataset & Preprocessing**
- *Objective:* Data flow.
- Bullets: 12 classes, ~243k samples each; window 1824 (256 for LLM); stratified split; test always 100 % real; **caveat: normalization differs across modules**.
- *Diagram:* raw signal → windows → normalize.
- *Notes:* "I window the long signals, normalize, and always keep a real-only test set. One honest caveat: normalization isn't yet unified across modules — on my fix-list."

**Slide 8 — Methodology**
- *Objective:* The generate→check→utility loop.
- Bullets: 4 generators → spectral check (Pearson/cosine/KL/MMD) → BigTCN utility check.
- *Diagram:* Section 3 master pipeline.
- *Notes:* "Every generator is judged twice: does its spectrum match, and does it actually help a classifier."

**Slide 9 — Four Generation Strategies**
- *Objective:* Contrast the generators.
- Bullets: GPT-4.1 prompt (zero-shot baseline); GPT-2 LoRA (text-serialized); local 17.4M transformer (recon + 47 features + stats matching); WGAN-GP (class-conditional, primary).
- *Diagram:* 4-column comparison.
- *Notes:* "From cheapest/weakest — prompting a frontier LLM — to the most engineered, a class-conditional GAN. The GAN is where most of the work went."

**Slide 10 — WGAN-GP Architecture Evolution**
- *Objective:* Show the 11-step trajectory.
- Bullets: transformer → CNN → TCN → +ACGAN class-conditioning (4→10→12 classes) → +self-attention → stability hardening.
- *Diagram:* timeline of the 11 models (Section 4.1 table).
- *Notes:* "I iterated the generator eleven times. The arc: move to temporal convolutions for efficiency, add class conditioning via an auxiliary classifier critic, then add self-attention for long-range spectral structure."

**Slide 11 — Final Model Detail**
- *Objective:* The current generator/critic.
- Bullets: Generator = label-embed → upsample → 9 Attention-TCN blocks (dilated) → global self-attention → tanh; Critic = TCN with **dual heads** (Wasserstein + 12-class); loss = WGAN-GP (λ=10) + classification (+FFT losses in standalone).
- *Diagram:* G/D block diagram.
- *Notes:* "Self-attention inside every dilated TCN block captures long-range periodicity; the critic doubles as a 12-way classifier, which both conditions generation and stabilizes it."

**Slide 12 — Training Strategy**
- *Objective:* Losses & optimization.
- Bullets: Wasserstein + gradient penalty + ACGAN CE; FFT magnitude/phase/PSD losses; AdamW/Adam; gradient clipping; spectral-norm/pre-norm for stability. **No contrastive/SSL.**
- *Diagram:* loss-term stacked bar with weights (from `v6.md`).
- *Notes:* "Because physical correctness lives in the spectrum, I add explicit frequency-domain losses on top of the adversarial objective."

**Slide 13 — Experimental Setup**
- *Objective:* Reproducibility.
- Bullets: SEED=42, 10 repeats; SEQ_LEN 1824; batch 32–64; 25–50 epochs; CUDA; **gaps: GPU unrecorded, no env lock**.
- *Diagram:* config table.
- *Notes:* "Everything is seeded and repeated ten times. I'm transparent that environment pinning and hardware logging are still missing."

**Slide 14 — Results (headline)**
- *Objective:* The key number.
- Bullets: accuracy 98.5 %→98.6 %(30 %)→95.2 %(50 %)→84.7 %(90 % synthetic); N/7IR/7BA = 100 %; 14BA collapses first.
- *Diagram:* accuracy-vs-synthetic% curve (Section 8 table / `total_based_gen_mix_train90.png`).
- *Notes:* "The punchline: I can replace a third of real data with synthetic at zero cost, and accuracy degrades gracefully after that — except one stubborn class, 14-mil ball faults."

**Slide 15 — Qualitative Examples**
- *Objective:* Show fidelity.
- Bullets: real vs synthetic FFT per class with Pearson; time-domain overlays; per-generator FFT comparisons.
- *Diagram:* `fft_comparison_best_columns.png` + `train{30,50,90}_gen_mix.png`.
- *Notes:* "Spectra line up on most classes; you can literally see where 14BA's peaks are under-reproduced."

**Slide 16 — Challenges**
- *Objective:* Honest difficulties.
- Bullets: GAN instability (hence heavy regularization); normalization inconsistency; class-specific failures; brittle GPT-2 numeric decoding; reproducibility/repo-hygiene.
- *Diagram:* risk heat-grid (technical vs research).
- *Notes:* "The engineering fight was stability; the scientific fight is per-class fidelity and domain shift across loads."

**Slide 17 — Current Status & Future Work**
- *Objective:* Where we are / next.
- Bullets: done = full GAN→classifier loop + one result table; next = unify preprocessing, benchmark all four generators on one classifier, fix 14BA, report FID/F1 numerically.
- *Diagram:* completed/partial/not-started kanban.
- *Notes:* "The WGAN-GP line is mature. The immediate win is a single fair benchmark of all four generators and fixing the weak classes."

**Slide 18 — Key Takeaways & Discussion**
- *Objective:* Close + invite questions.
- Bullets: synthetic vibration data is a viable ~⅓ substitute; class-conditional TCN+self-attention GAN is the strongest of four; frequency-domain fidelity is the right bar; open: domain adaptation, fair multi-generator benchmark.
- *Diagram:* one-line contribution summary.
- *Notes:* "Bottom line: synthetic data meaningfully eases label scarcity for bearing diagnosis, the GAN wins, and my next milestone is a clean four-way benchmark."

---

## 16. Technical Timeline (chronological, from file dates & git) [V]

| Period | Milestone |
|--------|-----------|
| Jun 2025 | GPT-4.1 prompting + GPT-2 LoRA fine-tuning baselines; first local-LLM versions (v1–v5) |
| Jul 2025 | Local-LLM v6/v7/v7.1 (17.4M transformer), epoch sweeps, FID tracking, ONNX export |
| Aug 2025 | WGAN-GP started: transformer → CNN generators; FFT/feature-matching/STFT losses; ACGAN class-conditioning introduced |
| Aug–Sep 2025 | Full-TCN conditional GANs (4→10→12 classes); attention-type ablation harness |
| Sep 2025 | `self attention/` synthetic CSVs committed; classifier mix-study (`model_v1_with_generated_data`, `train*_gen_mix.png`) |
| Oct 2025 | Hybrid GAN (spectral-norm/pre-norm/SE) + hybrid synthetic CSVs; total-replacement experiment committed |
| Jan 2026 | Deep TCN+transformer generator; **`model_self_attention.py` (current)**; classifier v1/v2 + `model_0.8train.py`; `full_pipeline.py` end-to-end |

---

## 17. 20 Likely Professor Questions (with answers)

1. **Why bearing fault diagnosis / CWRU?** [V] Standard, well-characterized benchmark with physical fault frequencies; lets us test spectral fidelity rigorously and compare to literature.
2. **Why is synthetic data needed if you have CWRU?** [I] CWRU is a proof-of-concept; the real target is industrial settings where labelled fault data is scarce/imbalanced — we use CWRU to validate the methodology.
3. **Why a TCN classifier rather than a CNN/transformer?** [V/I] TCNs give large receptive fields via dilation with stable, cheap training on long 1-D signals; it's a fixed, fair yardstick for all generators.
4. **Why WGAN-GP over vanilla GAN?** [V] Wasserstein + gradient penalty gives smoother gradients and far better training stability for 1-D signals; empirically the whole project leaned into stability.
5. **Why an ACGAN-style critic (auxiliary classifier)?** [V] Class-conditional generation for 12 fault types and the classification loss regularizes/conditions the generator.
6. **Why add explicit FFT/PSD/phase losses?** [V] Bearing faults are defined by spectral peaks; MSE in time domain alone doesn't guarantee correct frequencies, so we constrain the spectrum directly.
7. **Why self-attention inside TCN blocks?** [I] To capture long-range periodic structure (harmonics) that pure local convolutions miss; the architecture sweep converged on it.
8. **What's the headline quantitative result?** [V] Replacing real with synthetic: 98.5 %→98.6 % (30 %)→95.2 % (50 %)→84.7 % (90 %); ~⅓ substitution is free.
9. **Which classes fail and why?** [V/I] 14BA (and 7OR at extreme mixes); the generator under-reproduces those spectra — a per-class fidelity gap, not global.
10. **Is the generator memorizing or generating?** [V for LLM] The local-LLM is explicitly seed-reconstruction (risk of copying); the WGAN-GP samples from noise (less so) — but we haven't yet measured novelty/diversity quantitatively. [U]
11. **Why four generation strategies?** [V] To compare cost/quality trade-offs: zero-shot prompting (cheapest) → fine-tuning → custom transformer → GAN (most engineered).
12. **Which strategy is best?** [I] WGAN-GP self-attention — it's the only one wired into the quantitative classifier loop and the most developed; a fair four-way benchmark is the pending step. [U]
13. **Why does the local-LLM need 47 hand-engineered features?** [V] To inject time/frequency/nonlinear priors and enforce statistical matching (mean/std/skew/kurtosis/range) the raw model wouldn't learn from 256 samples alone.
14. **Any contrastive / self-supervised pretraining?** [V] No — "self-attention" is architectural; supervision is reconstruction + statistical + adversarial. (Honest scope clarification.)
15. **How do you evaluate generation quality besides accuracy?** [V] FFT-domain Pearson, cosine, KL divergence, MMD per class (best-matching column), plus FID curves; numeric tables are a pending deliverable. [U]
16. **What's the biggest weakness right now?** [I] Inconsistent preprocessing across modules + only one committed quantitative result → cross-experiment comparisons aren't yet airtight.
17. **What's the current bottleneck?** [I] GAN training stability and per-class fidelity (14BA), plus the lack of a unified config to run all generators identically.
18. **Does it generalize across operating loads/RPM?** [V] `model_v2.py` tests train-on-loads→test-on-held-out-load (4-class, confusion matrices); results not yet committed numerically. [U]
19. **Hardware / cost?** [I/U] CUDA single-GPU; `bitsandbytes` suggests 8-bit work to fit LLMs; exact GPU not logged.
20. **What's the next milestone?** [I] Unify preprocessing → benchmark all four generators through one classifier → numeric FID/F1 tables → fix weak classes.

---

## 18. Final Assessment

**Accomplished [V]:** A coherent research program with (a) four working synthetic-signal generators, (b) a matured class-conditional **TCN + self-attention WGAN-GP** (11 documented iterations), (c) FFT-based spectral evaluation tooling, and (d) a downstream-utility harness producing the key, defensible finding that **synthetic vibration data can replace ~30–40 % of real training data with no accuracy loss**, degrading gracefully thereafter.

**Remaining [V/U]:** Unify preprocessing/config; benchmark all four generators through the *same* classifier; commit numeric FID/spectral/F1 tables; fix per-class generation failures (14BA); add reproducibility scaffolding (README, requirements/Docker, weights as artifacts).

**Biggest technical risk [I]:** The headline conclusion rests on a **single generator family and a single committed results table**, atop **inconsistent preprocessing** — so the "synthetic data works" claim is real but not yet robustly generalized or fairly compared across the four strategies.

**Highest-impact next step [I]:** One unified, seeded benchmark that runs all four generators through the identical BigTCN classifier and reports downstream accuracy + FID in one table — turning four exploratory lines into a single, citable comparison.
```
