🚀 ISL Zero-Latency Real-Time Translator — Implementation Plan
📌 Project Overview
An end-to-end, client-side Indian Sign Language (ISL) real-time translator. The system extracts 234 geometric landmarks from a webcam feed using MediaPipe JS (WebGL) and runs inference locally using ONNX Runtime Web, achieving sub-15ms latency with 0ms network lag.

📁 Target Directory Layout
Plaintext
D:\ISL_1_PROJECT\isl_recognition\
│
├── IMPLEMENTATION_PLAN.md     # Master Architecture & Execution Plan
├── index.html                 # Unified Frontend (Modules 1–6)
├── isl_model.onnx             # Exported ONNX Machine Learning Pipeline
├── export_model.py            # Python ONNX Export Script
│
└── processed_data/
    ├── label_map.json         # Class ID mapping (WEAR, WELCOME, WHAT, WHERE, WHO, WORRY, YOU)
    ├── isl_model.pkl          # Scikit-Learn Model Artifact
    └── scaler.pkl             # StandardScaler Artifact
🏗️ Architecture Pipeline
Plaintext
[Webcam Stream] ──> [MediaPipe Holistic JS (WebGL)]
                           │
                           ▼ (Extract Pose, Left & Right Hand Keypoints)
               [234-Feature Normalization Engine]
                           │
                           ▼ (Float32Array[234])
             [ONNX Runtime Web (Client GPU/CPU)]
                           │
                           ▼ (Predicted Class ID)
             [Debounce & Sentence Builder]
                           │
                           ├──> [UI / Glassmorphism HUD Overlay]
                           └──> [Browser Web Speech TTS Engine]
🧩 Module Breakdown & Vibe Coding Execution Blueprint
Module 1: Camera Stream & Video Pipeline
Goal: Initialize high-framerate webcam capture, mirror video feed horizontally, and sync an overlaid canvas for dynamic drawing.

Specifications:

Request webcam at 640x480 resolution using navigator.mediaDevices.getUserMedia.

Display mirrored video using CSS (transform: scaleX(-1)).

Align a transparent HTML5 <canvas> directly on top of the <video> element.

Handle camera permissions and graceful error fallback.

🤖 AI Vibe-Coding Prompt (Module 1)
"Create a lightweight HTML5 camera module inside index.html. Request webcam access at 640x480 resolution, mirror the video display horizontally using CSS, and place an absolute-positioned transparent <canvas> directly over the video with matching dimensions. Add state handling for missing camera hardware or denied permissions."

Module 2: 234-Feature Normalization Engine (JavaScript)
Goal: Port Python landmark normalization logic into pure JS to convert MediaPipe outputs into a fixed Float32Array(234).

Specifications:

Pose (33 x 3 = 99): Normalized relative to shoulder center and divided by shoulder width.

Left Hand (21 x 3 = 63): Normalized relative to wrist coordinate and divided by hand span.

Right Hand (21 x 3 = 63): Normalized relative to wrist coordinate and divided by hand span.

Hand-to-Nose Distances (3 + 3 = 6): Left wrist to nose, Right wrist to nose (scaled by shoulder width).

Wrist-to-Wrist Distance (3): Relative distance between left and right wrists (scaled by shoulder width).

Zero-Padding: Return 0-filled arrays if hands/pose are unobserved.

🤖 AI Vibe-Coding Prompt (Module 2)
"Write a standalone JavaScript function extract234Features(results) that takes a MediaPipe Holistic results object and produces a 234-element Float32Array. Ensure exact math parity with Python: normalize pose to shoulder width and center, normalize left/right hands to wrist position and hand span, calculate hand-to-nose relative offsets, calculate wrist-to-wrist relative distance, and apply zero-padding when hands are not detected."

Module 3: In-Browser ONNX Runtime Web Execution
Goal: Load isl_model.onnx and label_map.json to execute local model predictions on every MediaPipe frame tick.

Specifications:

Load ONNX Runtime Web via CDN (ort.min.js).

Initialize an ort.InferenceSession pointing to ./isl_model.onnx.

Load ./processed_data/label_map.json and build a reverse ID-to-Word lookup object.

Wrap the 234-feature array into an ort.Tensor('float32', data, [1, 234]).

Run session.run({ float_input: tensor }) and extract predicted class labels.

🤖 AI Vibe-Coding Prompt (Module 3)
"Integrate ONNX Runtime Web (ort.min.js) in index.html. Write an async function initONNX() to load ./isl_model.onnx and fetch ./processed_data/label_map.json. Create a prediction function predictSign(features234) that builds an input tensor of shape [1, 234], runs session.run(), and returns the predicted sign word string with maximum probability."

Module 4: UI/UX & Live Translation Glassmorphism HUD
Goal: Build a dark-mode interface with video viewports, live skeleton rendering, and real-time status indicators.

Specifications:

Theme: Slate dark background (#0f172a), neon blue accents (#38bdf8), green success (#22c55e).

Floating Glassmorphism HUD showing Detected Sign, Confidence/Status, and FPS Counter.

Canvas skeleton rendering: Green lines (#22c55e) for Left Hand, Yellow lines (#eab308) for Right Hand.

Badges for active states: "LIVE SYSTEM ACTIVE", "Waiting for Hands...", or "Loading Model...".

🤖 AI Vibe-Coding Prompt (Module 4)
"Build a modern glassmorphism UI layout in index.html using CSS. Style a central video card with rounded corners, a floating HUD overlay showing current translated word and status badge, and custom canvas drawing loops for green/yellow MediaPipe hand skeletons. Make the interface responsive and clean."

Module 5: Sentence Assembly & Debounce Engine
Goal: Eliminate prediction noise and flickering by requiring sustained gestures before committing words to a sentence bar.

Specifications:

Frame Debouncing: Require a word to remain consistent for 8–10 consecutive frame ticks (~150–200ms) before committing.

Sentence Array: Maintain an array of confirmed words (e.g., ["WELCOME", "YOU"]).

UI Controls: Display assembled text in a dedicated bar with Clear, Backspace, and Copy buttons.

🤖 AI Vibe-Coding Prompt (Module 5)
"Write a JavaScript sentence debouncer module. Add logic to track incoming frame predictions so a sign is only added to an array if detected consistently for 10 consecutive frames. Create UI elements to display the assembled sentence, along with buttons to clear the sentence or delete the last added word."

Module 6: Text-to-Speech (TTS) & Accessibility Engine
Goal: Convert single predicted signs or assembled sentences into speech output using the native Web Speech API.

Specifications:

Use window.speechSynthesis and SpeechSynthesisUtterance.

Add a "🔊 Speak Sentence" button.

Add an optional "Auto-Speak" toggle switch to pronounce words immediately upon confirmation.

Configure speech parameters (Rate: 0.9, Pitch: 1.0, Language: en-US / en-IN).

🤖 AI Vibe-Coding Prompt (Module 6)
"Write a Web Speech API module in JavaScript. Implement functions speakText(text) and speakCurrentSentence(). Add controls for speech playback and integrate an Auto-Speak toggle button that triggers audio whenever a new debounced word is added to the sentence array."

🛠️ Verification Checklist
[ ] Run python export_model.py and confirm isl_model.onnx is created at the root directory.

[ ] Open index.html via static server (python -m http.server 8000 or Netlify/Vercel).

[ ] Open Browser DevTools (F12) and confirm console outputs: ✅ ONNX Model Loaded Successfully!.

[ ] Perform gestures (WEAR, WELCOME, WHAT, WHERE, WHO, WORRY, YOU) in front of the camera.

[ ] Confirm hand skeleton overlays draw smoothly at 60 FPS.

[ ] Confirm the HUD updates translation results in under 15ms.