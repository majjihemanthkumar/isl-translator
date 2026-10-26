# Indian Sign Language (ISL) Real-Time Translator

> Zero-latency, in-browser Indian Sign Language interpretation powered by MediaPipe Holistic, ONNX Runtime Web (WASM), and Groq LLM natural language refinement.

---

## 📖 Complete Documentation
For an in-depth technical explanation covering the mathematics, normalization engine, 234-feature extraction, model training, and module architecture, please refer to:
👉 **[PROJECT_EXPLANATION.md](file:///d:/old_version_isl/PROJECT_EXPLANATION.md)**

For a concise summary of the technologies and tools used:
👉 **[TECH_STACK.txt](file:///d:/old_version_isl/TECH_STACK.txt)**

---

## ⚡ Quick Start

### 1. Web Application (Recommended)
Double-click **`run.bat`** (or execute `.\run.bat` in your terminal).
- Starts the local Python web server on port `8000`.
- Automatically opens `http://localhost:8000` in your default browser.
- Uses WebAssembly for client-side zero-latency inference.

### 2. Desktop Mode (OpenCV)
Double-click **`run_desktop.bat`** (or execute `python live_testing.py`).
- Opens a native desktop OpenCV window with live skeletal tracking and prediction HUD.

---

## 🏗️ System Overview
1. **Module 1 (`camera.js`):** 60 FPS video capture and hardware-accelerated canvas sync.
2. **Module 2 (`feature_extractor.js`):** Derives an invariant 234-element normalized feature vector from 543 MediaPipe landmarks.
3. **Module 3 (`onnx_engine.js`):** Client-side ONNX Runtime Web inference running `isl_model.onnx` via WASM.
4. **Module 4 (`index.html` & `styles.css`):** Glassmorphic dark-mode dashboard with real-time telemetry.
5. **Module 5 (`sentence_engine.js`):** Multi-frame temporal debouncing (8 consecutive frames) to assemble sign glosses.
6. **Module 6 (`ai_translator.js`):** Translates telegraphic sign glosses into fluent English using Groq LLMs (Qwen/LLaMA) with Text-To-Speech (TTS).

---

## 🏷️ Supported Signs
`ABUSE`, `AGREE`, `ALOT`, `WANT`, `WATER`, `WEAR`, `WELCOME`, `WHAT`, `WHERE`, `WHO`, `WORRY`.
