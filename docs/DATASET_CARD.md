# Dataset Card: CREMA-D, RAVDESS, and Contact-Center Dialog Benchmarks

## 1. Overview & Motivation
This project builds and evaluates Speech Emotion Recognition (SER) and Conversation Analytics systems using publicly accessible, license-clear emotional speech corpora because restricted corpora (IEMOCAP, MSP-Podcast) require institutional data-transfer agreements unavailable within the project window.

| Corpus | Role | Raw Clips | Valid Clips | Speakers | Sampling Rate | License |
|---|---|---:|---:|---:|---:|---|
| **CREMA-D** | Primary training, validation & speaker-independent test | 7,442 | 7,441 | 91 (diverse age/sex/race) | 16,000 Hz PCM | ODbL v1.0 / DbCL v1.0 |
| **RAVDESS** (Speech) | Zero-shot cross-corpus evaluation | 1,440 | 1,440 | 24 (12 F, 12 M) | 48,000 Hz PCM (resampled to 16 kHz) | CC BY-NC-SA 4.0 |
| **CREMA-Dialog** | Constructed multi-turn contact-center benchmark | 380 dialogs | 380 dialogs | 91 (partition-locked) | 16,000 Hz | Derived from CREMA-D (same ODbL/DbCL terms) |
| **RAVDESS-Dialog** | Cross-corpus conversation benchmark | 40 test dialogs | 40 test dialogs | 24 | 16,000 Hz | Derived from RAVDESS (CC BY-NC-SA 4.0) |

## 2. Provenance & Acquisition
- **CREMA-D**: Acquired programmatically via `research/scripts/download_data.py` and `research/scripts/fetch_crema_meta.py` from `https://github.com/CheyneyComputerScience/CREMA-D` (Cao et al., 2014, *IEEE Trans. Affective Computing* 5(4):377–390). Includes `VideoDemographics.csv` and crowd-sourced voice-only voting files (`processedResults/summaryTable.csv`, `tabulatedVotes.csv`).
- **RAVDESS**: Acquired via `research/scripts/download_data.py` from Zenodo record `1188976` (`Audio_Speech_Actors_01-24.zip`; Livingstone & Russo, 2018, *PLoS ONE* 13(5):e0196391).

## 3. Data Validation & Duplicate Audit
Executed automatically by `research/src/ser_research/data_pipeline.py` (outputs recorded in [dataset_stats.json](file:///C:/Users/saransh%20sinha/Documents/Speech%20Emotion%20Recognition%20and%20Conversation%20Analytics%20Tool/results/dataset_stats.json)):
1. **Quality & Silence Checks**:
   - CREMA-D: 7,442 WAV files scanned; **7,441 passed validation**; 1 silent/corrupted file (`1076_MTI_SAD_XX.wav`, peak=0.00000, silence_ratio=1.00) was automatically detected and excluded. Mean raw duration: `2.543 s`; mean estimated SNR: `20.21 dB`.
   - RAVDESS: 1,440 WAV files scanned; **1,440 valid** (0 excluded). Mean raw duration: `3.701 s`; mean estimated SNR: `67.37 dB` (clean studio recordings).
2. **Duplicate Audit**:
   - Exact SHA-256 byte-duplicate count across valid CREMA-D WAVs: `3`.
   - Near-duplicate count (64-dim acoustic functional cosine similarity > 0.99995): `4`.
3. **Human Listener Agreement**:
   - Joined voice-only crowd-sourced evaluations from `tabulatedVotes.csv` (rows `100001..107442`). Mean voice-only agreement across CREMA-D is `0.637`, allowing stratification into Low (`< 0.50`), Medium (`0.50–0.75`), and High (`>= 0.75`) perceptual ambiguity bins.

## 4. Label Harmonization (No Blind Merging)
- **T6 (Primary CREMA-D Task)**: `{NEU: 1087, HAP: 1271, SAD: 1270, ANG: 1271, FEA: 1271, DIS: 1271}`.
- **T4 (Core 4-Class Cross-Corpus)**: `{NEU, HAP, SAD, ANG}` shared between CREMA-D and RAVDESS (`672` RAVDESS utterances).
- **T6_CROSS (Shared 6-Class Cross-Corpus)**: `{NEU, HAP, SAD, ANG, FEA, DIS}` (`1,056` RAVDESS utterances). RAVDESS `calm` and `surprised` are **excluded** from cross-corpus evaluation rather than forcibly merged.

## 5. Leakage-Safe Speaker-Independent Partitioning
Speakers never cross split boundaries (`speaker_overlap_assertion = PASSED`, verified by automated test `test_speaker_leakage_zero_overlap_if_built`):
- **CREMA-D (91 actors)** stratified by `(Sex, AgeBucket)`:
  - `train`: **61 actors**, **4,983 utterances**
  - `val`: **15 actors**, **1,230 utterances**
  - `test`: **15 actors**, **1,228 utterances**
- Standardization parameters (Log-Mel mean/std and 64-dim functional scalers) are fit **strictly on the `train` partition**.

## 6. Multi-Turn Dialog & Stream Benchmarks
- **CREMA-Dialog (`380` dialogues; `240 train / 60 val / 80 test`)**: Each conversation pairs one Customer actor and one Agent actor drawn strictly from the same speaker partition across 5 contact-center archetypes (`escalating_gradual`, `escalating_abrupt`, `resolved`, `routine_neutral`, `chronic_negative`). Escalation onset $\tau$ is computed via the formal baseline-vs-sustained-distress definition $(W=3, \rho=1/3, k=3, m=2)$, yielding **28 escalating** and **52 non-escalating** test dialogues.
- **Stream Benchmark (`stream_benchmark.npz`)**: `285` continuous 10.24-second (`64 x 512`) multi-turn spectrograms with frame-level segmentation masks and interval bounding boxes used for the semantic segmentation and object-detection benchmark.
- **Demo Stereo Audio (`data/conversations/sample_*.wav`)**: Three realistic stereo WAV recordings (`Customer` left channel, `Agent` right channel) synthesized strictly from held-out `test` partition actors.

## 7. Known Limitations
- Both CREMA-D and RAVDESS contain **acted** North American English speech over fixed scripted sentences rather than spontaneous telephony calls.
- Because `CREMA-Dialog` composes real acted utterances into turn sequences rather than recording live interactive dialogue, co-articulation and real-time overlap between speakers are absent; context findings are explicitly framed within this constructed benchmark limit.
- All emotion labels represent *perceived/expressed vocal expression*, never internal psychological state.
