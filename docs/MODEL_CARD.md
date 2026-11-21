# Model Card: ResNet-SER & Causal Context Transformer (v1.0.0)

## 1. Model Details & Registry Checksums
| Artifact | File | Architecture | Opset | SHA-256 |
|---|---|---|---|---|
| **Utterance Acoustic Backbone** | `models/resnet_ser.onnx` | 2D Residual CNN (`stride=2` stem + `MaxPool2d` + 4 ResBlocks `24->48->96` + stats pooling `192->128` embedding) | 17 | `ed2c3139a018e9baf15d2625a60afad68fd01020a0a4858dcab138ed583ea422` |
| **Conversational Sequence Head** | `models/context_transformer.onnx` | Causal 2-layer Transformer (`d=128`, 4 heads, sinusoidal PE, speaker one-hot + 128-d embedding) | 17 | `9e30de126f651d5b6f9b3be40fbd8e805e6b8da79342ab859d196c3118b59b67` |

- **Classes (`T6`)**: `["NEU", "HAP", "SAD", "ANG", "FEA", "DIS"]`.
- **Input Specification**: Normalized `[B, 1, 64, 128]` Log-Mel spectrogram (`sr=16,000 Hz`, `n_fft=512`, `hop=320`, 64 HTK Mel bins `50..7600 Hz`, standardized by train mean `-5.468255` and std `3.428258`).
- **Calibration**: Post-hoc Temperature Scaling $T^* = 1.2564$ fitted on the speaker-independent validation split (`N=1,230`).

---

## 2. Intended Use & Out-of-Scope Applications
- **Intended Use**: Academic SER research and **assisted post-call contact-center quality triage** (ranking calls by probable customer vocal frustration buildup for human supervisor review).
- **Out-of-Scope / Prohibited Uses**:
  - Scoring, monitoring, disciplining, or evaluating contact-center **employees/agents** (prohibited under **EU AI Act Article 5(1)(f)**; blocked by default in `ConversationAnalyticsEngine`).
  - Diagnosing mental health, depression, personality traits, or internal "true" psychological feelings.
  - Automated financial, legal, or punitive customer decisions without human review.

---

## 3. Quantitative Evaluation (Speaker-Independent CREMA-D Test, `N=1,228`, 15 Unseen Speakers)

### 3.1 Utterance Classification & Calibration
| Metric | `ResNet-SER` (No SpecAug) | `ResNet-SER + SpecAug` (Exported) | `ResNet-SER + SpecAug` (T4 4-Class) |
|---|---:|---:|---:|
| **Accuracy (WA)** | 54.40% | 53.75% | 69.06% |
| **Unweighted Avg Recall (UAR)** | 54.75% | 53.80% ± 0.55% | 69.67% |
| **Macro-F1** | 54.00% | 53.04% | 68.92% |
| **ECE (Uncalibrated -> Calibrated)** | — | 0.1378 -> **0.0708** | — |

### 3.2 Demographic Subgroup Fairness (`ResNet-SER + SpecAug`, `T=1.256`, CREMA-D Test)
| Attribute | Subgroup | Support (`n`) | UAR | Macro-F1 |
|---|---|---:|---:|---:|
| **Sex** | Female | 572 | 56.79% | 55.68% |
| | Male | 656 | 52.21% | 50.87% |
| **Age** | 20–34 | 654 | 55.35% | 54.17% |
| | 35–49 | 328 | 47.07% | 45.50% |
| | 50+ | 246 | 61.38% | 59.15% |
| **Race** | African American | 410 | 52.74% | 51.07% |
| | Asian | 164 | 56.25% | 54.62% |
| | Caucasian | 654 | 54.89% | 54.30% |

*Observed fairness spread*: 4.58 pp UAR gap across Sex (Female > Male), 14.31 pp gap across Age brackets (`35–49` lowest at 47.07%), and 3.51 pp across Race groups (`African American` 52.74% vs `Asian` 56.25%).

### 3.3 Cross-Corpus Generalization & Acoustic Robustness
| Condition | UAR | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| **In-Corpus Clean (Studio 16 kHz)** | 54.35% | 53.07% | 53.75% |
| **Additive Noise SNR 20 dB** | 52.58% | 51.30% | 52.04% |
| **Additive Noise SNR 10 dB** | 29.29% | 21.88% | 30.05% |
| **Additive Noise SNR 5 dB** | 17.30% | 6.46% | 17.75% |
| **8 kHz Telephony Bandlimit** | 35.85% | 29.97% | 36.73% |
| **Zero-Shot Cross-Corpus (`RAVDESS T6`, `N=1,056`)** | 30.90% | 24.83% | 31.72% |
| **Zero-Shot Cross-Corpus (`RAVDESS T4`, `N=672`)** | 39.32% | 33.49% | 41.52% |

---

## 4. Known Failure Modes & Guidance
1. **Context Transformer 6-Class Collapse**: While `context_transformer.onnx`'s **escalation head (`D6`)** achieves the highest turn-level escalation ranking (`AUROC = 0.8030`, `AUPRC = 0.4004`), its **6-class emotion logits** trained on 240 dialogues suffer overfitting/collapse (`UAR = 27.20%` vs `ResNet-SER` `51.62%` and `HMM-Forward` `66.45%`). Production analytics should rely on calibrated `ResNet-SER` + `HMM-Forward` smoothing for utterance emotion labels and use `D6` / `CUSUM` strictly for escalation triage.
2. **Noise & Codec Degradation**: Performance drops sharply below 15 dB SNR or under 8 kHz telephony bandwidth (`-18.5 pp` UAR), requiring telephony-band augmentation before live call-center deployment.
