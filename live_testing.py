import os
import json
import cv2
import joblib
import numpy as np
import mediapipe as mp

# --- 1. PATH RESOLUTION & ARTIFACT LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

MODEL_PATH = os.path.join(PROCESSED_DIR, "isl_model.pkl")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.pkl")
MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. Run train_model.py first!")

print("=" * 65)
print(f"📂 Loading trained artifacts from: {PROCESSED_DIR}")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(MAP_PATH, "r") as f:
    label_map = json.load(f)

# Create reverse lookup: class index -> word string
id_to_word = {v: k for k, v in label_map.items()}
print(f"✅ Model loaded for {len(label_map)} classes: {', '.join(label_map.keys())}")
print("=" * 65)

# --- 2. MEDIAPIPE INITIALIZATION ---
try:
    from mediapipe.python.solutions import holistic as mp_holistic
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except ImportError:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

def extract_234_features(results):
    """Extracts 234 normalized relative keypoint coordinates."""
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    
    l_shoulder, r_shoulder = pose[11], pose[12]
    shoulder_center = (l_shoulder + r_shoulder) / 2.0
    shoulder_width = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
    nose = pose[0]
    
    pose_rel = (pose - shoulder_center) / shoulder_width
    
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        lh_wrist = lh[0]
        lh_span = np.linalg.norm(lh[9] - lh[0]) + 1e-6
        lh_rel = (lh - lh_wrist) / lh_span
        lh_pos_rel = (lh_wrist - nose) / shoulder_width
    else:
        lh_rel, lh_pos_rel = np.zeros((21, 3)), np.zeros(3)
        
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        rh_wrist = rh[0]
        rh_span = np.linalg.norm(rh[9] - rh[0]) + 1e-6
        rh_rel = (rh - rh_wrist) / rh_span
        rh_pos_rel = (rh_wrist - nose) / shoulder_width
    else:
        rh_rel, rh_pos_rel = np.zeros((21, 3)), np.zeros(3)
        
    w2w_rel = (lh[0] - rh[0]) / shoulder_width if (results.left_hand_landmarks and results.right_hand_landmarks) else np.zeros(3)
    
    return np.concatenate([
        pose_rel.flatten(),
        lh_rel.flatten(),
        rh_rel.flatten(),
        lh_pos_rel,
        rh_pos_rel,
        w2w_rel
    ])

# --- 3. LIVE CAMERA LOOP ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not access webcam. Check device connection.")
    exit(1)

print("\n🎥 LIVE WEBCAM INTERPRETER RUNNING")
print("👉 Press 'q' inside the video window to exit.\n")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        # Render skeleton keypoints
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
        # Inference Logic
        if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            features = extract_234_features(results).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            probs = model.predict_proba(features_scaled)[0]
            pred_id = np.argmax(probs)
            confidence = probs[pred_id]
            pred_word = id_to_word.get(pred_id, "Unknown")
            
            # Highlight green if confidence > 50%, orange if uncertain
            color = (0, 255, 0) if confidence > 0.50 else (0, 165, 255)
            
            # HUD Overlay
            cv2.rectangle(frame, (15, 15), (420, 95), (0, 0, 0), -1)
            cv2.rectangle(frame, (15, 15), (420, 95), color, 2)
            
            cv2.putText(frame, f"SIGN: {pred_word}", (30, 52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (30, 82), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (15, 15), (420, 60), (0, 0, 0), -1)
            cv2.putText(frame, "Waiting for gestures...", (30, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("ISL 7-Word Real-Time Interpreter", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("👋 Live testing window closed successfully.")