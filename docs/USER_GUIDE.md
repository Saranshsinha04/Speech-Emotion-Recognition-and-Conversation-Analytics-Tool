# User Guide: Speech Emotion Recognition & Conversation Analytics Tool

## 1. Prerequisites
- **Java**: JDK 21 (`C:\Program Files\Java\jdk-21`)
- **Maven**: Bundled project wrapper `.\mvn.cmd` (uses `tools\apache-maven-3.9.9` and offline `.m2repo/`)
- **Python (for research scripts & tests only)**: Python 3.13 (`C:\Python313\python.exe`) with `torch`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `onnx`, `onnxscript`, `onnxruntime`

---

## 2. Quickstart Commands

### Run All Automated Tests
- **Java unit, gradient-check, DSP parity, ONNX parity, and integration tests**:
  ```powershell
  .\mvn.cmd -o test
  ```
- **Python dataset-leakage, DSP feature, and model shape tests**:
  ```powershell
  C:\Python313\python.exe -m unittest discover -s research\tests -v
  ```

### Run the Primary Java Application (`--demo` CLI Mode)
Builds the shaded jar (if not already built) and runs both the `ser-math` mathematical sandbox and the `ser-app` ONNX conversation analytics demo on the 3 held-out contact-center stereo recordings:
```powershell
.\mvn.cmd -o package
java -jar ser-app\target\ser-app.jar --demo
```
Outputs:
- Mathematical sandbox SVG charts & summary: `results\math_sandbox\`
- Structured JSON analytics reports per conversation: `results\demo_reports\`

### Analyze Any Custom `.wav` Recording
```powershell
java -jar ser-app\target\ser-app.jar --analyze path\to\conversation.wav
```
- **Channel convention**: For stereo 16-bit PCM WAV files, Channel 0 (Left) is mapped to `CUSTOMER` and Channel 1 (Right) to `AGENT`. Mono files are segmented into `CUSTOMER` utterances via energy VAD.
- **Compliance note**: By default, employee/agent emotion labels are redacted (`REDACTED_POLICY`) per EU AI Act Art. 5(1)(f). For offline corpus research only, pass `--research-mode`.

### Launch the Interactive Web Dashboard
```powershell
java -jar ser-app\target\ser-app.jar --serve --port 8080
```
Open `http://127.0.0.1:8080` in a browser to access:
1. **Dashboard Overview**: Processed calls, escalation rate, average RTF, and emotion distribution.
2. **Conversation Analysis**: Play audio, inspect turn-by-turn acoustic vs contextual predictions, view HMM escalation & CUSUM curves, transition matrices, or upload a new `.wav` file.
3. **Research & Benchmarks**: Live tables and SVG visualizations from `results\experiment_results.json`.
4. **Math Sandbox & Paradigms**: Live outputs from `ser-math`.
5. **Responsible AI & Go/No-Go**: Policy constraints and deployment guidance.

---

## 3. Reproducing the Research Pipeline from Scratch
1. Download metadata & build tool (requires network):
   `python research\scripts\download_data.py && python research\scripts\fetch_crema_meta.py`
2. Build leakage-safe dataset manifests, cached features, and dialog benchmarks:
   `$env:PYTHONPATH='research\src'; C:\Python313\python.exe -m ser_research.data_pipeline`
3. Run all deep learning benchmarks, ablations, escalation detectors, and ONNX exports:
   `$env:PYTHONPATH='research\src'; C:\Python313\python.exe -u -m ser_research.run_all_experiments`
