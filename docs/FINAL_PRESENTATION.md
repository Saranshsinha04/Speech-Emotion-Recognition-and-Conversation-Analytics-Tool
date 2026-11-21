# Academic & Industry Presentation Deck
## Speech Emotion Recognition & Conversation Analytics Tool for Contact Centers

---

### Slide 1: Title & Project Scope
- **Title:** Speech Emotion Recognition & Conversation Analytics Tool for Contact Centers
- **Domain:** Contact-Center Quality Triage & Affective Speech Analytics
- **Stack:** Pure-Java Mathematical Sandbox (`ser-math`), Java 21 + ONNX Runtime (`ser-app`), PyTorch Experimentation (`research/`), Maven 3.9.9
- **Central Question:** Can combining acoustic CNNs with conversational context improve expressed vocal emotion estimation, detect frustration escalation, and run efficiently in production Java?

---

### Slide 2: Industry Motivation & Regulatory Boundaries
- Contact-center QA teams manually audit a tiny fraction of support calls; supervisors need call-level escalation triage rather than noisy clip-level labels.
- **EU AI Act Article 5(1)(f):** Prohibits inferring employee emotions in the workplace.
- **Design Control:** Product defaults to **Customer-Only Policy Mode** (`REDACTED_POLICY` on agent channel); outputs are strictly framed as calibrated estimates of *expressed vocal emotion*.

---

### Slide 3: Dataset Audit & Speaker-Disjoint Engineering
- **CREMA-D (Primary):** `7,442` scanned → **`7,441` valid** (`1076_MTI_SAD_XX.wav` silent file excluded), `3` exact SHA-256 duplicates, `4` near-duplicates audited; **`61 train (4,983) / 15 val (1,230) / 15 test (1,228)`** speakers (**0 actor overlap**).
- **RAVDESS (Zero-Shot Cross-Corpus):** `1,440` valid clips across `24` actors (`48 kHz -> 16 kHz` resampled; `1,056` shared T6, `672` T4).
- **Constructed Benchmarks:** `CREMA-Dialog` (`380` multi-turn calls; `80` test: `28` escalating / `52` non-escalating) and `stream_benchmark` (`285` continuous `10.24 s` streams).

---

### Slide 4: Java Mathematical Foundations (`ser-math`)
- Zero-dependency pure-Java library implementing N-D `Tensor` broadcasting, `matmul`, `conv2d`, `conv3d`, receptive fields, and one-sided Jacobi `Svd`.
- **Automated Numerical Gradient Checking ($\epsilon = 10^{-5}$):** Passed all 91 parameters across Sigmoid, Tanh, ReLU, BatchNorm1D, and L2 regularization (`maxRelErr = 2.753e-8`).
- **Real CREMA-D Baseline:** Scratch Java MLP (`17,158` params, Nesterov + BN + Dropout) trained on 64 functionals reached **`50.71%` UAR** (`50.49%` Acc) on the 15 unseen test speakers.

---

### Slide 5: Classical Neural Learning Paradigms
- Isolated comparative sandbox (`LearningParadigms.java`):
  - **Hebbian vs Oja's Rule:** Pure Hebbian norm grew to `1.999e8`; Oja's normalization maintained `||w|| = 1.0001` and aligned with the exact SVD 1st principal direction (`cosine = 0.99990`).
  - **Competitive Learning (1D SOM):** Quantization error reduced `1.1195 -> 0.4537`.
  - **Delta / LMS Error-Correction:** MSE converged `1.6356 -> 2.09e-15`.
  - **Energy-Based (RBM CD-1 & Hopfield):** RBM reconstruction MSE `0.2537 -> 0.0840`; Hopfield energy minimized `-1.0 -> -3.0` (`100%` pattern recall).

---

### Slide 6: E1 & E2 — Acoustic Representations & CNN/ResNet Benchmarks
- **E1 Features (`CNN-Basic`):** `Log-Mel` (`46.79%` UAR) > `Linear Mel` (`45.60%`) > `MFCC+Delta` (`45.28%`) > `STFT` (`44.26%`).
- **E2 Architectures (`CREMA-D 6-Class`, `N=1,228`):**
  - `MLP-Baseline`: `48.17%` UAR | `CNN-Basic`: `47.86%` | `Conv3D-SER`: `47.16%` | `GoogLeNet-SER`: `51.55%`
  - `ResNet-SER`: **`54.75%`** UAR (`54.00%` F1) | `CNN-Deep`: **`54.88%`** UAR
  - `ResNet-SER + SpecAug` (**4-class T4**): **`69.67%` UAR** (`69.06%` Acc)

---

### Slide 7: E3, E4 & E12 — Optimizers, Regularization & Calibration
- **Optimizers (MLP):** `AdamW` (`51.62%`) ≈ `Momentum` (`51.58%`) ≈ `Nesterov` (`51.16%`) > `SGD` (`45.80%`).
- **Temperature Scaling ($T^* = 1.256$):** Reduced `ResNet-SER + SpecAug` Expected Calibration Error from **`0.1378` to `0.0708`** (`-48.6%`).
- **Visual Explainability:** Grad-CAM highlights harmonic pitch contours and high-energy formant transitions driving Anger/Disgust predictions.

---

### Slide 8: Item 8 Benchmark — Semantic Segmentation vs Temporal Event Detection
Benchmarked on `60` continuous `10.24 s` multi-turn call streams (`64 x 512`):
- **Semantic Segmentation (`SpecUNet-1D`):** `74.89%` Frame Accuracy, `0.5682` mIoU (`4.37 ms`).
- **Two-Stage Detection:** `R-CNN-1D` (`0.2628` mAP@0.5, `2.77 ms`), `Fast-RCNN-1D` (`0.2531` mAP@0.5, `3.24 ms`), `Faster-RCNN-1D` (`0.2898` mAP@0.5, best localization IoU **`0.7918`**, `5.09 ms`).
- **Single-Stage Detection:** `YOLO-1D` (`0.3151` mAP@0.5, `2.83 ms`), **`RetinaNet-1D` (FPN + Focal Loss)** (**`0.4294` mAP@0.5**, **`0.5336` Event F1**, `4.43 ms`).

---

### Slide 9: E14 — Generative Augmentation (CVAE & cGAN) & Latent Space
- **CVAE & cGAN:** Raw generated spectrograms achieved `17.0%–17.1%` classifier label consistency due to over-smoothed spectral detail.
- **Downstream Regularization Win:** Mixing `900` CVAE synthetic samples into `ResNet-SER` training boosted 6-class UAR to **`56.29%`** (`55.14%` Macro-F1) and achieved the best calibration (**`ECE = 0.0298`**, Brier `0.5846`).
- PCA latent projections show well-clustered arousal/valence geometry in ResNet 128-d space vs compact Gaussian CVAE latent space.

---

### Slide 10: Stage C — Conversational Context & Hypothesis H2
Evaluated on `CREMA-Dialog` (`80` test calls) using speaker-disjoint cross-fitted embeddings:
- `No-Context (ResNet-SER)`: `51.62%` UAR (`69.00%` SeqAcc, `0.1309` JSD)
- **`HMM-Forward (Non-Neural Control)`:** **`66.45%` UAR** (`+14.83 pp` overall; **`+14.94 pp`** on Low-Agreement ambiguous turns, `74.24%` SeqAcc, `0.0816` JSD)
- `BiLSTM`: `37.48% ± 0.40%` | `Causal Transformer`: `27.20%` (bootstrap CI vs No-Ctx `[-0.322, -0.204]`)
- **Scientific Finding:** Temporal context strongly resolves ambiguous turns via Markov smoothing, but unanchored neural sequence models overfit small `240`-dialog corpora.

---

### Slide 11: E11 — Frustration Escalation Detection & Sensitivity
- Operational onset $\tau$: calm baseline ($\le 1/3$ negative over `3` turns) followed by sustained distress ($\ge 2/3$ negative).
- **Detector Comparison:**
  - `D6 TransformerHead` achieved top ranking (**`AUROC = 0.8030`**, **`AUPRC = 0.4004`**, median delay `+1.0` turn).
  - `D4 CUSUM (Audio-Only CNN)` achieved top event-level F1 (**`0.5538`**, lowest FA/track `0.365`).
- Sensitivity grid across $(W, \rho, k, m)$ confirms prevalence scales predictably (`51.25% -> 35.0% -> 17.5%`).

---

### Slide 12: E9 & E10 & E13 — Fairness, Cross-Corpus & Telephony Robustness
- **Subgroup Fairness (Sex/Age/Race):** Female `56.79%` vs Male `52.21%`; Age `50+: 61.38%` vs `35-49: 47.07%`; Race spread `3.51 pp` (`African American 52.74%`, `Caucasian 54.89%`, `Asian 56.25%`).
- **Cross-Corpus (`CREMA-D -> RAVDESS`):** Zero-shot UAR drops to `30.90%` (`T6`) / `39.32%` (`T4`).
- **Noise & Telephony:** Stable at `20 dB SNR` (`52.58%` vs `54.35%` clean), drops to `35.85%` under `8 kHz` telephony bandlimit and `29.29%` at `10 dB SNR`.

---

### Slide 13: Production Java System & Parity Verification
- Single shaded executable (`ser-app/target/ser-app.jar`) with CLI (`--demo`, `--analyze`) and REST + Web Dashboard (`--serve`).
- **Parity Verified:** Pure-Java `AudioFeatures` matches Python DSP (`< 5e-3` Log-Mel diff across 10 real test WAVs); Java ONNX Runtime matches PyTorch (`< 1e-4`; ORT `2.78e-6` ResNet, `1.19e-6` Transformer).
- **Real-Time Speed:** End-to-end Java call processing `239–715 ms` on `28–38 s` stereo calls (**`RTF = 0.0085–0.0242`**).

---

### Slide 14: Failure Case Analysis from Live Demo Calls
- **`sample_routine_neutral.wav`:** `ResNet-SER` predicted `5/5` customer turns accurately (`NEU, HAP, NEU, NEU, NEU`) and `D6_TransformerHead` stayed low (`<= 0.276`), while the collapsed 6-class `ContextTransformer` logits misled the HMM filter (`0.917`).
- **Lesson:** Production pipelines must feed `HMM-Forward` / `CUSUM` with calibrated `ResNet-SER` posteriors, pairing them with `D6`'s escalation head.

---

### Slide 15: Hypothesis Verdict Scorecard
- **H1 (CNN/ResNet > MLP):** **Confirmed** (`54.75%–56.29%` vs `48.17%` / `50.71%`).
- **H2 (Context > No-Context & HMM):** **Partially Confirmed / Neural Refuted** (`HMM-Forward` `66.45%` beats No-Context `51.62%` by `+14.8 pp`, proving context value, but `Causal TF` `27.20%` fails to beat `HMM` on 240 dialogs).
- **H3 (Sequential Escalation > Threshold):** **Confirmed** (`D6 AUROC 0.8030 / AUPRC 0.4004` & `D4 EventF1 0.5538` beat `D1`).
- **H4 (Augmentation Helps):** **Confirmed for CVAE calibration/UAR (`56.29%`, ECE `0.0298`)**; cross-corpus gap remains large.
- **H5 (Java/ONNX RTF <= 0.1 & Parity <= 1e-4):** **Confirmed** (`RTF <= 0.024`, parity `< 1e-4`).

---

### Slide 16: Industry Go / No-Go Deployment Recommendation
- **Assisted Post-Call Quality Triage (Customer Channel):** **CONDITIONAL GO** — fast (`RTF < 0.025`), well-calibrated (`ECE = 0.0298`), strong escalation ranking (`AUROC = 0.803`), contingent on wiring CUSUM/HMM to `ResNet-SER` posteriors and retraining with `8 kHz` telephony noise augmentation.
- **Unsupervised Real-Time Routing / Auto-Actions:** **NO-GO** (false alarms & `< 10 dB` / `8 kHz` sensitivity).
- **Agent / Employee Emotion Scoring:** **PROHIBITED (NO-GO)** per EU AI Act Art. 5(1)(f).
