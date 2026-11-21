# Technical Research Report
## Speech Emotion Recognition and Conversation Analytics Tool for Contact Centers

**Authoring System:** Research & Production Engineering Pipeline (Java 21 + PyTorch + ONNX Runtime)  
**Primary Corpora:** CREMA-D (7,441 valid utterances, 91 actors) & RAVDESS (1,440 utterances, 24 actors)  
**Date:** November 21, 2026

---

## Abstract
Conventional Speech Emotion Recognition (SER) systems classify isolated short clips under speaker-leaky evaluation protocols and lack formal definitions for conversation-level phenomena such as customer frustration escalation. This project investigates whether combining acoustic CNN representations with conversational context improves expressed vocal emotion recognition, detects frustration escalation, and supports practical Java/ONNX deployment for contact centers. We construct a speaker-independent benchmark on CREMA-D (`61/15/15` speaker split, 0 actor overlap), cross-corpus zero-shot evaluation on RAVDESS, a 380-dialogue multi-turn benchmark (`CREMA-Dialog`), and a 285-stream continuous segmentation/detection benchmark (`stream_benchmark`). On utterance classification, `ResNet-SER` (`54.75%` UAR) and `CNN-Deep` (`54.88%` UAR) outperform hand-crafted 64-feature MLPs (`48.17%` PyTorch / `50.71%` Java), and `ResNet-SER` augmented with a Conditional VAE reaches `56.29%` UAR and `0.0298` ECE (`69.67%` UAR on 4-class T4). On continuous multi-turn streams, single-stage `RetinaNet-1D` with Focal Loss (`0.4294` mAP@0.5, `4.43 ms`) outperforms `YOLO-1D` (`0.3151`) and `Faster-RCNN-1D` (`0.2898`), while `SpecUNet-1D` achieves `74.89%` frame accuracy. Crucially, rigorous Hypothesis H2 ablation refutes the assumption that deep neural sequence heads automatically beat simple probabilistic controls: an `HMM-Forward` temporal filter boosts `CREMA-Dialog` UAR from `51.62%` to **`66.45%`** (`+14.94 pp` on low-agreement ambiguous turns), whereas unanchored `Causal Transformer` (`27.20%`) and `BiLSTM` (`37.88%`) models trained on 240 dialogues suffer 6-class overfitting—even as the Causal Transformer's binary escalation head (`D6`) achieves top escalation ranking (`AUROC = 0.8030`, `AUPRC = 0.4004`). Finally, the complete Java 21 / ONNX Runtime product achieves `< 1e-4` numerical parity (`2.78e-6`), `RTF = 0.0085–0.0242` end-to-end on real stereo audio, and enforces EU AI Act Art. 5(1)(f) customer-only defaults.

---

## 1. Research Objective, Industry Motivation & Hypotheses

### 1.1 Central Research Question
> *"Can combining acoustic speech emotion recognition with conversational context improve emotion classification, identify customer frustration escalation, and provide more reliable conversation-level analytics than conventional audio-only emotion recognition?"*

### 1.2 Industry Context & Regulatory Framing
Contact centers review only a small fraction of recorded interactions manually. Supervisors need calibrated triage signals—distinguishing an isolated raised-voice utterance from sustained customer frustration buildup—without deploying pseudo-scientific claims about "internal psychological state." Under **EU AI Act Article 5(1)(f)**, AI systems inferring emotions of employees in the workplace are prohibited; therefore, our production architecture defaults strictly to **Customer-Channel-Only** analytics (`customerOnlyPolicyMode = true`).

### 1.3 Literature Context
- **Acoustic SER**: OpenSMILE/GeMAPS functional baselines (Schuller et al. 2009; Eyben et al. 2016) established engineered prosodic/spectral baselines; spectrogram CNNs and attention pooling (Satt et al. 2017; Mirsamadi et al. 2017) replaced frame-by-frame hand features; self-supervised speech encoders (wav2vec 2.0, WavLM, Chen et al. 2022) represent the large-model ceiling (~95M–300M params), which are infeasible to fine-tune locally on 4 GB VRAM without cloud scale.
- **Emotion Recognition in Conversation (ERC)**: bc-LSTM (Poria et al. 2017), CMN (Hazarika et al. 2018), and DialogueRNN (Majumder et al. 2019) demonstrate context gains primarily using *textual transcripts*. Whether acoustic-only context outperforms non-neural temporal smoothing under strict cross-fitted embeddings remains a critical open test.
- **Change-Point & Calibration**: Page's (1954) CUSUM, 2-state HMM filtering (Rabiner 1989), SpecAugment (Park et al. 2019), and Temperature Scaling (Guo et al. 2017) anchor our escalation and uncertainty design.

### 1.4 Formal Research Hypotheses
- **H1**: Spectrogram 2D CNN/ResNet architectures outperform an MLP on utterance-level acoustic functionals for speaker-independent SER (UAR).
- **H2**: Temporal context modelling over cross-fitted utterance embeddings improves UAR over (a) no-context CNN and (b) non-neural HMM forward smoothing, especially on low-agreement ambiguous turns.
- **H3**: Causal escalation detectors over contextual/sequential posteriors outperform naive per-utterance thresholding on AUPRC and detection delay.
- **H4**: Augmentation (SpecAugment / CVAE) improves generalization and calibration under speaker/corpus shift.
- **H5**: The Java 21 / ONNX Runtime pipeline achieves `RTF <= 0.1` and `< 1e-4` PyTorch↔Java inference parity.

---

## 2. Dataset Engineering & Leakage-Safe Splits
Full audit artifacts are stored in [dataset_stats.json](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/results/dataset_stats.json) and [DATASET_CARD.md](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/DATASET_CARD.md).
- **CREMA-D**: `7,442` WAV files scanned (`16 kHz`); `7,441` valid (`1076_MTI_SAD_XX.wav` excluded: peak=0.0, silence=1.00); `3` exact SHA-256 duplicates and `4` cosine `> 0.99995` near-duplicates audited; stratified speaker-independent split across `91` actors into **`61 train` (`4,983`) / `15 val` (`1,230`) / `15 test` (`1,228`)** with automated `0`-overlap assertion. Mean voice-only crowd agreement = `0.637`.
- **RAVDESS**: `1,440` valid speech WAVs (`48 kHz -> 16 kHz` polyphase resampled, `24` actors). Cross-corpus evaluation uses strictly shared classes (`T4 = 672`, `T6 = 1,056`), excluding `calm` and `surprised`.
- **CREMA-Dialog & RAVDESS-Dialog**: `380` partition-locked multi-turn calls (`240 train / 60 val / 80 test`; `28/80` escalating under default $(W=3, \rho=1/3, k=3, m=2)$) and `40` cross-corpus RAVDESS dialogs (`13` escalating).

---

## 3. Mathematical Foundations & Learning Paradigms (`ser-math`)
Implemented from scratch in pure Java (`ser-math`, zero external runtime libraries):
- **Tensor Algebra & Convolutions**: Row-major N-D `Tensor` with NumPy broadcasting, 2D/batched `matmul`, inner/outer products, `conv2d`, `conv3d` over `[Log-Mel, Delta-T, Delta-F]`, and exact receptive field verification (`rf = 18` for 5-layer conv/pool stack).
- **Jacobi SVD**: Pure-Java one-sided Jacobi SVD reconstructing arbitrary rectangular matrices to `1e-10` precision; low-rank approximation of a `32x48` harmonic spectrogram reaches `7.17%` Frobenius error at rank 4 and `0.33%` at rank 8.
- **Automated Numerical Gradient Checking**: Central-difference checks ($\epsilon = 10^{-5}$) across Sigmoid, Tanh, and ReLU MLPs with BatchNorm and L2/L1 regularization achieved max relative error **`2.753e-8`** (`91` parameters, all passing `< 1e-6`).
- **Optimizers (`Batch GD`, `SGD`, `Momentum`, `Nesterov`)**: Verified convergence on synthetic 4-class acoustic data (`100%` val acc for SGD/Momentum/Nesterov, `98.33%` for Batch GD) and trained a `17,158`-parameter Java MLP directly on the real `4,983`-utterance CREMA-D 64-dim functionals (`Nesterov + BatchNorm + Dropout=0.25`), reaching **`50.71%` UAR (`50.49%` Acc, `50.52%` Macro-F1)** on the unseen 15-speaker test set.
- **Isolated Learning Paradigms (`LearningParadigms.java`)**:
  1. *Hebbian* ($\Delta w = \eta y x$): norm exploded to `1.999e8` without normalization.
  2. *Modified Hebbian / Oja's Rule* ($\Delta w = \eta y (x - yw)$): maintained unit norm (`1.0001`) and converged to **`0.99990`** cosine alignment with the exact SVD 1st principal eigenvector.
  3. *Competitive Learning (1D SOM)*: quantization error dropped `1.1195 -> 0.4537`.
  4. *Error-Correction (Widrow-Hoff LMS Delta Rule)*: MSE converged `1.6356 -> 2.09e-15`.
  5. *Energy-Based (RBM CD-1 & Hopfield)*: RBM reconstruction MSE dropped `0.2537 -> 0.0840`; Hopfield energy minimized `-1.0 -> -3.0` with `100%` pattern recovery from 25% bit corruption.

---

## 4. Utterance SER Benchmarks (Experiments E1–E4, E12, E14)

### 4.1 E1: Acoustic Feature Representation (`CNN-Basic`, CREMA-D Test `N=1,228`)
| Feature Representation | Accuracy | UAR | Macro-F1 | Train Time |
|---|---:|---:|---:|---:|
| **Log-Mel (`64x128`)** | **46.25%** | **46.79%** | 43.57% | 43.7s |
| Linear Mel (`64x128`) | 44.87% | 45.60% | 42.39% | 36.6s |
| MFCC + Delta (`40x128`) | 45.03% | 45.28% | **43.89%** | 23.4s |
| STFT Magnitude (`64x128`) | 44.06% | 44.26% | 40.77% | 36.2s |

### 4.2 E2 & E14: Architecture & Augmentation Benchmark (`CREMA-D` Speaker-Independent)
| Model | Params | Acc | UAR | Macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| `MLP-Baseline` (PyTorch, 64 functionals) | 17,350 | 47.64% | 48.17% | 47.49% | 0.0371 |
| `Java Scratch MLP` (Nesterov + BN) | 17,158 | 50.49% | 50.71% | 50.52% | — |
| `CNN-Basic` (2 seeds) | 32,614 | 47.39% | 47.86% ± 0.78% | 45.86% | 0.0470 |
| `Conv3D-SER` (`[LogMel, dT, dF]`) | — | 46.7% | 47.16% | 44.75% | — |
| `GoogLeNet-SER` (Inception 1x1/3x3/5x5) | — | 51.22% | 51.55% | 50.58% | 0.0363 |
| `ResNet-SER + SpecAug` (2 seeds, exported) | — | 53.75% | 53.80% ± 0.55% | 53.04% | 0.1378 -> **0.0708** ($T^*=1.256$) |
| `ResNet-SER` (No SpecAug) | — | 54.40% | 54.75% | 54.00% | — |
| `CNN-Deep` (VGG double-conv blocks) | 140,694 | 54.32% | 54.88% | 53.69% | 0.0900 |
| **`ResNet-SER + CVAE Aug` (E14)** | — | **55.70%** | **56.29%** | **55.14%** | **0.0298** (Brier `0.5846`) |
| `ResNet-SER + SpecAug` (**T4 4-Class Core**) | — | **69.06%** | **69.67%** | **68.92%** | — |

**Findings:** **H1 is confirmed** (`ResNet-SER` / `CNN-Deep` / `ResNet+CVAE` exceed both PyTorch and Java MLPs by `+4.0 to +8.1 pp` UAR). `Conv3D-SER` (`47.16%`) underperformed 2D architectures because frequency-time-delta depth lacks true volumetric translation invariance. In **E3 Optimizers**, `AdamW` (`51.62%`), `Momentum` (`51.58%`), and `Nesterov` (`51.16%`) beat plain `SGD` (`45.80%`) by `> +5 pp`. In **E14 Generative Augmentation**, although raw CVAE/cGAN decoded spectrograms had only `17.0–17.1%` independent classifier consistency (smoothing fine spectral harmonics), adding `900` CVAE synthetic samples as soft spectral regularizers improved downstream `ResNet-SER` UAR to **`56.29%`** and lowered ECE to **`0.0298`**.

---

## 5. Item 8 Benchmark: Classification, Semantic Segmentation & Detection (`E8-SegDet`)
Evaluated on `60` continuous `10.24-second` multi-turn test streams (`64 x 512` frames, `3` classes: Silence, Non-Negative Speech, Negative Speech):

| Model | Paradigm | Frame Acc / mIoU | mAP@0.5 | Event F1 | Mean IoU | Latency (ms/stream) |
|---|---|---:|---:|---:|---:|---:|
| `SpecUNet-1D` | Semantic Segmentation | **74.89%** / **0.5682** | 0.1732 | 0.3036 | 0.7544 | 4.37 ms |
| `R-CNN-1D` | External Proposals + Crop CNN | — | 0.2628 | 0.4236 | 0.7851 | 2.77 ms |
| `Fast-RCNN-1D` | Shared Backbone + RoI-Align | — | 0.2531 | 0.4346 | 0.7553 | 3.24 ms |
| `Faster-RCNN-1D` | Learned Anchor RPN + RoI-Align | — | 0.2898 | 0.4780 | **0.7918** | 5.09 ms |
| `YOLO-1D` | 16-Cell Grid Single-Shot | — | 0.3151 | 0.4766 | 0.7645 | **2.83 ms** |
| **`RetinaNet-1D`** | **2-Scale FPN + Focal Loss ($\gamma=2$)** | — | **0.4294** | **0.5336** | 0.7493 | 4.43 ms |

**Architectural Synthesis:** `Faster-RCNN-1D` produces the tightest localized boundaries (`0.7918` IoU), `YOLO-1D` runs fastest (`2.83 ms`), and `RetinaNet-1D`'s Focal Loss handles class imbalance between silence frames and emotional speech turns best (`0.4294` mAP@0.5, `+11.4 pp` over YOLO-1D). For utterance classification after segmentation, standard 2D ResNets replace inappropriate spatial bounding-box detectors.

---

## 6. Stage C: Conversational Context & Hypothesis H2 Analysis (`CREMA-Dialog`)
Evaluated using 2-fold speaker-disjoint **cross-fitted** `ResNet` embeddings (`N=80` test dialogues) to prevent optimistic overfitting on training utterances:

| Model | UAR | Macro-F1 | Seq Acc | Edit Sim | JSD (↓) | Low-Agr UAR (`<0.50`) | High-Agr UAR (`>=0.75`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `No-Context (ResNet-SER)` | 51.62% | 50.47% | 69.00% | 69.30% | 0.1309 | 47.92% | 57.19% |
| **`HMM-Forward (Non-Neural Control)`** | **66.45%** | **66.80%** | **74.24%** | **74.52%** | **0.0816** | **62.86%** (+14.94) | **70.60%** (+13.41) |
| `ResNet + Causal LSTM` | 33.16% | 32.61% | 45.92% | 47.45% | 0.1486 | 25.49% | 32.29% |
| `ResNet + BiLSTM` (2 seeds) | 37.88% (`±0.40%`) | 36.98% | 54.90% | 55.55% | 0.1294 | 33.12% | 31.01% |
| `ResNet + Causal Transformer` (3 seeds) | 27.20% (`±0.95%`) | 20.67% | 27.54% | 30.10% | 0.3150 | 22.69% | 32.38% |
| `ResNet + Bidir Transformer` (3 seeds) | 28.81% (`±1.03%`) | 23.35% | 30.58% | 35.02% | 0.2105 | 29.21% | 31.34% |

- **Cluster-Bootstrap 95% CI (`Causal TF - No-Context`)**: `[-0.3215, -0.2037]`; vs `HMM-Forward`: `[-0.4812, -0.3435]`.
- **Context Window Ablation (`Causal TF`)**: `K=1: 30.00%`, `K=3: 34.80%`, `K=5: 35.88%`, `K=8: 34.50%`, `K=Full: 26.40%` (longer windows exacerbate overfitting on `240` dialogues).
- **Critical Negative Finding (H2 Outcome):** While **temporal context strongly helps** (`HMM-Forward` raises UAR by **+14.83 pp** overall and **+14.94 pp** on low-agreement ambiguous turns, cutting conversation JSD from `0.1309` to `0.0816`), **unconstrained neural sequence heads trained from scratch on only 240 dialogues overfit and underperform both the no-context CNN and the non-neural HMM control**. This validates why including the HMM baseline in Phase 1 was essential: without it, sequence engineering complexity would not have been exposed as counterproductive on small-N conversation data.

---

## 7. Escalation Detection Benchmark (`E11`) & Failure Analysis

| Detector | Source | AUROC | AUPRC | Event F1 | Median Delay | FA / Non-Esc Track |
|---|---|---:|---:|---:|---:|---:|
| `D1 Threshold` | Audio-Only CNN | 0.7344 | 0.3518 | 0.4565 | -1.0 turns | 0.827 |
| **`D4 CUSUM`** | **Audio-Only CNN** | 0.7591 | 0.3770 | **0.5538** | +2.0 turns | **0.365** |
| `D1 Threshold` | Context TF Probs | 0.4593 | 0.1865 | 0.1176 | -2.0 turns | 1.000 |
| `D4 CUSUM` | Context TF Probs | 0.6071 | 0.2665 | 0.4368 | -1.0 turns | 0.769 |
| `D5 HMM Filter` | Context TF Probs | 0.6740 | 0.2941 | 0.4646 | -1.0 turns | 0.923 |
| **`D6 TransformerHead`** | **Learned Sequence** | **0.8030** | **0.4004** | 0.4776 | **+1.0 turns** | 0.442 |

**Live Java Failure-Case Diagnosis (`results/demo_reports/*.json`)**:
Inspecting the three stereo test WAVs run through `ser-app.jar --demo` confirms the exact mechanistic link between E6 and E11:
- On `sample_routine_neutral.wav` (ground truth customer turns `[NEU, HAP, NEU, NEU, NEU]`), the acoustic `ResNet-SER` predicted `[NEU, HAP, NEU, NEU, NEU]` (**5/5 exact!**), and `D6_TransformerHead` correctly stayed quiet (`peak=0.2755 < 0.45`). However, the overfitted 6-class `ContextTransformer` head drifted those utterances toward `[SAD, DIS, DIS, DIS, DIS]`, causing the downstream HMM escalation filter (which read `ContextTransformer` 6-class posteriors) to raise a false alert (`0.9174`).
- Conversely, on `sample_escalating_gradual.wav`, `D6_TransformerHead` spiked cleanly at Turn 8 (`0.5320 >= 0.45`), catching the escalation buildup!
- **Actionable Architecture Lesson**: Production conversation analytics must wire the `HMM-Forward` / `CUSUM` escalation detector directly to **calibrated `ResNet-SER` (or `HMM-smoothed ResNet-SER`) posteriors** and pair them with `D6`'s learned escalation score, discarding the collapsed 6-class neural sequence logits.

---

## 8. Cross-Corpus, Fairness, Robustness & Production Efficiency (`E9, E10, E13, H5`)
- **Cross-Corpus (`CREMA-D -> RAVDESS`, E10)**: Zero-shot UAR drops to `30.90%` (`T6`) and `39.32%` (`T4`), highlighting studio/speaker-style domain shift.
- **Demographic Fairness (E9)**: Sex (`Female 56.79%` vs `Male 52.21%`), Age (`20–34: 55.35%`, `35–49: 47.07%`, `50+: 61.38%`), Race (`African American 52.74%`, `Asian 56.25%`, `Caucasian 54.89%`).
- **Noise & Telephony Robustness (E13)**: Graceful degradation at `20 dB SNR` (`52.58%` vs `54.35%` clean), sharp collapse at `10 dB` (`29.29%`), and `-18.5 pp` drop under `8 kHz` telephony bandlimiting (`35.85%`).
- **Deployment Parity & Speed (H5 Confirmed)**:
  - PyTorch↔ONNX max abs diff: **`2.78e-6`** (`ResNet`), **`1.19e-6`** (`Transformer`).
  - Java `AudioFeatures` vs Python `features.py` Log-Mel parity: `< 5e-3` across all 10 real test WAVs; Java ORT vs Python parity: **`< 1e-4`**.
  - Java end-to-end call processing (`VAD + DSP + ResNet + Context + Analytics`): **`239.3–715.0 ms`** for **`28.1–38.1 s`** calls (**`RTF = 0.00853–0.02419`**, well under the `0.1` budget).

---

## 9. Evidence-Based Industry Go / No-Go Deployment Recommendation

| Use Case | Verdict | Justification from Measured Evidence |
|---|---|---|
| **Offline Post-Call Quality Review Triage (Customer Channel)** | **CONDITIONAL GO** | Fast CPU execution (`RTF <= 0.024`), well-calibrated `ResNet-SER + CVAE` (`ECE = 0.0298`), strong binary escalation ranking (`D6 AUROC = 0.8030`), and `HMM-Forward` smoothing (`66.45%` UAR) provide reliable priority sorting for human QA reviewers—**provided** (1) escalation CUSUM/HMM is wired to calibrated `ResNet-SER` posteriors rather than the collapsed 6-class Transformer head, and (2) models are fine-tuned with 8 kHz telephony codec & noise augmentation on domain calls. |
| **Live Unsupervised Routing / Escalation Auto-Actions** | **NO-GO** | False-alarm rates (`0.365–0.442` per non-escalating track) and domain shift under 8 kHz telephony (`35.85%` UAR) and `<10 dB` noise (`29.29%`) remain too high for autonomous decisions without human verification. |
| **Employee / Agent Emotion Scoring** | **PROHIBITED (NO-GO)** | Explicitly prohibited in workplace settings under **EU AI Act Article 5(1)(f)** and outside scientific validity. |
