import os
import json
import cv2
import warnings
import numpy as np
import joblib
import mediapipe as mp

from mediapipe.python.solutions import holistic as mp_holistic
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# --- 1. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Automatically detect ALL word folders in datasets/
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"❌ Dataset directory not found at: {DATASET_DIR}")

TARGET_WORDS = sorted([
    d for d in os.listdir(DATASET_DIR) 
    if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')
])

if not TARGET_WORDS:
    raise RuntimeError(f"❌ No word folders found inside: {DATASET_DIR}")

print("=" * 65)
print(f"🚀 DYNAMIC ISL TRAINING PIPELINE")
print(f"📂 Dataset Location : {DATASET_DIR}")
print(f"🏷️  Detected Words ({len(TARGET_WORDS)}) : {', '.join(TARGET_WORDS)}")
print("=" * 65)

# --- 2. MEDIAPIPE & FEATURE EXTRACTION ---
# mp_holistic = mp.solutions.holistic

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

# --- 3. SCAN FOLDERS AND EXTRACT LANDMARKS ---
X_list = []
y_list = []

label_map = {word: idx for idx, word in enumerate(TARGET_WORDS)}
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

holistic = mp_holistic.Holistic(
    static_image_mode=True, 
    model_complexity=1, 
    min_detection_confidence=0.5
)

print("\n🔍 Extracting 234 landmark features per image...")

for word in TARGET_WORDS:
    folder_path = os.path.join(DATASET_DIR, word)
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) 
        if f.lower().endswith(valid_extensions)
    ]
    
    print(f"   • Class '{word}': Found {len(image_files)} images...")
    
    class_id = label_map[word]
    processed_count = 0
    
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            features = extract_234_features(results)
            X_list.append(features)
            y_list.append(class_id)
            processed_count += 1

    print(f"     ✅ Successfully extracted keypoints from {processed_count}/{len(image_files)} images.")

holistic.close()

if len(X_list) == 0:
    raise RuntimeError("❌ No features could be extracted from the images!")

X = np.array(X_list)
y = np.array(y_list)

print(f"\n📊 Extracted Dataset Shape: X={X.shape}, y={y.shape}")

# Save feature matrices
np.save(os.path.join(OUTPUT_DIR, "X_features.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)

# --- 4. 5-FOLD STRATIFIED CROSS-VALIDATION ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []
all_y_true = []
all_y_pred = []

print("\n⚡ Running 5-Fold Stratified Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = ExtraTreesClassifier(n_estimators=250, max_depth=20, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, preds)
    
    fold_accuracies.append(acc)
    all_y_true.extend(y_val)
    all_y_pred.extend(preds)
    
    print(f"   Fold {fold}: Accuracy = {acc * 100:.2f}%")

avg_acc = np.mean(fold_accuracies) * 100

print("\n" + "=" * 65)
print(f"🎯 OVERALL MODEL ACCURACY Across {len(TARGET_WORDS)} Classes: {avg_acc:.2f}%")
print("=" * 65)

# --- 5. PERFORMANCE REPORT ---
print("\n📋 Classification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=TARGET_WORDS))

print("🧩 Confusion Matrix:")
cm = confusion_matrix(all_y_true, all_y_pred)

title_col = "True \\ Pred"
header = f"{title_col:<15}" + "".join([f"{name:>10}" for name in TARGET_WORDS])
print("-" * len(header))
print(header)
print("-" * len(header))
for i, row in enumerate(cm):
    row_str = f"{TARGET_WORDS[i]:<15}" + "".join([f"{val:>10}" for val in row])
    print(row_str)
print("-" * len(header))

# --- 6. RETRAIN FINAL MODEL & SAVE CLEAN ARTIFACTS ---
print("\n💾 Retraining final model on full dataset...")

final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

final_model = ExtraTreesClassifier(n_estimators=250, max_depth=20, random_state=42)
final_model.fit(X_scaled, y)

# Clean, fresh production artifact names
joblib.dump(final_model, os.path.join(OUTPUT_DIR, "isl_model.pkl"))
joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

print("\n✅ SAVED FRESH MODEL ARTIFACTS TO 'processed_data/':")
print("   • processed_data/isl_model.pkl")
print("   • processed_data/scaler.pkl")
print("   • processed_data/label_map.json")