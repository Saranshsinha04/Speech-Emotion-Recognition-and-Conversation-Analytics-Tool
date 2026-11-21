# Speech Emotion Recognition & Conversation Analytics Tool for Contact Centers

A research-driven, end-to-end Java + PyTorch/ONNX deep learning system for **expressed vocal emotion estimation**, **acoustic event segmentation/detection**, **conversational context modelling**, and **customer frustration escalation analytics** in contact centers.

> **Responsible-Use Notice:** All predictions estimate *expressed vocal emotion* as perceived by human annotators—never internal psychological state or personality. In alignment with **EU AI Act Article 5(1)(f)** (workplace emotion recognition ban), the Java application defaults to **Customer-Only Policy Mode**, redacting employee/agent vocal emotion scoring unless explicitly run in offline `--research-mode`.

---

## Single-Command Quickstart

### 1. Build & Run All Automated Tests
- **Java Unit, Gradient-Check, DSP Parity, ONNX Parity & Integration Tests** (11 tests across `ser-math` and `ser-app`):
  ```powershell
  .\mvn.cmd -o test
  ```
- **Python Dataset-Leakage, Feature-DSP & Model Tests** (4 tests):
  ```powershell
  C:\Python313\python.exe -m unittest discover -s research\tests -v
  ```

### 2. Run the Primary Java Application (`--demo` or Web Dashboard)
Build the self-contained executable jar and run the end-to-end mathematical sandbox + Java ONNX conversation analytics demo:
```powershell
.\mvn.cmd -o package
java -jar ser-app\target\ser-app.jar --demo
```
To launch the interactive Contact-Center Analytics & Research Web Dashboard on `http://127.0.0.1:8080`:
```powershell
java -jar ser-app\target\ser-app.jar --serve --port 8080
```
To analyze any 16-bit PCM WAV file via CLI:
```powershell
java -jar ser-app\target\ser-app.jar --analyze path\to\call.wav
```

---

## Key Verified Experimental Highlights

| Area | Metric / Finding | Measured Result |
|---|---|---|
| **Java Math Sandbox (`ser-math`)** | Numerical gradient check (Sigmoid/Tanh/ReLU + BatchNorm + L2) | Max rel error **`2.753e-8`** (`91` params, passed) |
| **Learning Paradigms (`ser-math`)** | Oja's rule vs Jacobi SVD 1st principal direction; Hopfield recall | Cosine **`0.99990`**; Hopfield **`100%`** |
| **Java Scratch MLP on CREMA-D** | 64-dim functionals, Nesterov + BN + Dropout (`17,158` params) | Test UAR **`50.71%`**, Macro-F1 **`50.52%`** |
| **CREMA-D 6-Class Speaker-Indep.** | `ResNet-SER` / `CNN-Deep` / `ResNet + CVAE Aug` (`N=1,228`) | UAR **`54.75%`** / **`54.88%`** / **`56.29%`** |
| **CREMA-D 4-Class Core (T4)** | `ResNet-SER + SpecAug` (`{NEU, HAP, SAD, ANG}`) | UAR **`69.67%`**, Acc **`69.06%`** |
| **Calibration (Temperature Scaling)** | Optimal $T^* = 1.256$ on `ResNet-SER + SpecAug` | ECE **`0.1378 -> 0.0708`** (`ResNet+CVAE`: **`0.0298`**) |
| **Seg & Detection (Item 8)** | `SpecUNet-1D` FrameAcc / `RetinaNet-1D` vs `YOLO-1D` vs `Faster-RCNN` | `SpecUNet` **`74.89%`** (`0.5682` mIoU); `RetinaNet-1D` **`0.4294` mAP@0.5** |
| **Stage C Context (H2 Control)** | `HMM-Forward` vs `No-Context` vs `BiLSTM` vs `Causal TF` | `HMM` **`66.45%`** UAR (+14.8 pp) > `No-Ctx` **`51.62%`** > `TF` **`27.20%`** |
| **Escalation Detection (E11)** | `D6 TransformerHead` vs `D4 CUSUM (Audio-Only CNN)` | `D6` **AUROC `0.8030` / AUPRC `0.4004`**; `D4` Event F1 **`0.5538`** |
| **Cross-Corpus & Telephony** | Zero-shot `CREMA-D -> RAVDESS` (T4); 8 kHz telephony drop | `RAVDESS` T4 UAR **`39.32%`**; 8 kHz UAR **`35.85%`** (vs `54.35%`) |
| **Java / ONNX Parity & RTF** | PyTorch↔Java ORT max diff; full WAV end-to-end RTF | Parity **`< 1e-4`** (`2.78e-6`); End-to-End **RTF `0.0085–0.0242`** |

---

## Repository Structure

```
├── pom.xml, mvn.cmd, .mvn/       # Java 21 + Maven 3.9.9 multi-module build (offline .m2repo supported)
├── ser-math/                     # Dependency-free Java tensor, SVD, neurons, MLP, optimizers, grad check, paradigms
├── ser-app/                      # Java WAV/VAD/DSP parity, ONNX Runtime service, analytics engine, REST + Web UI
├── research/
│   ├── scripts/                  # Reproducible CREMA-D / RAVDESS / Maven downloaders
│   ├── src/ser_research/         # features.py, data_pipeline.py, models.py, metrics.py, experiments_*.py, plots.py
│   └── tests/                    # Automated Python dataset-leakage, DSP & model tests
├── models/                       # Exported resnet_ser.onnx, context_transformer.onnx, registry.json, parity_fixtures.json
├── results/                      # Verified dataset_stats.json, experiment_results.json, SVG figures, demo_reports/
└── docs/                         # TECHNICAL_REPORT.md, DATASET_CARD.md, MODEL_CARD.md, ARCHITECTURE.md,
                                  # USER_GUIDE.md, FINAL_PRESENTATION.{md,html}, DEMO_WALKTHROUGH.html
```

---

## Documentation Index
- [Technical Research Report](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/TECHNICAL_REPORT.md)
- [Dataset Card](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/DATASET_CARD.md)
- [Model Card](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/MODEL_CARD.md)
- [Software Architecture](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/ARCHITECTURE.md)
- [User Guide](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/USER_GUIDE.md)
- [Final Presentation (Markdown)](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/FINAL_PRESENTATION.md) | [Slide Deck HTML](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/FINAL_PRESENTATION.html)
- [Interactive Demo Walkthrough](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/docs/DEMO_WALKTHROUGH.html)
