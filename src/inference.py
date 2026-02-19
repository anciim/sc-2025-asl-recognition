import os
import sys
import cv2
import numpy as np
import time
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_detection import create_hand_landmarker, detect_hand, detect_hand_fallback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = ["cat", "hello", "help", "more", "no",
               "please", "sorry", "thank_you", "what", "yes"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)


def load_cnn_model(model_path=None):
    if model_path is None:
        candidates = [
            os.path.join(MODELS_DIR, "cnn_model_final.keras"),
            os.path.join(MODELS_DIR, "cnn_model_finetuned.keras"),
            os.path.join(MODELS_DIR, "cnn_model.keras"),
            os.path.join(MODELS_DIR, "cnn_model_final.h5"),
            os.path.join(MODELS_DIR, "cnn_model_finetuned.h5"),
            os.path.join(MODELS_DIR, "cnn_model.h5"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)
        if model_path is None:
            print("GRESKA: Model nije pronadjen!")
            return None

    print(f"Ucitavanje modela: {model_path}")
    return load_model(model_path)


def predict_frame(model, frame, landmarker, confidence_threshold=0.3):
    hand_crop, landmarks, bbox = detect_hand(frame, landmarker, IMG_SIZE)

    if hand_crop is None:
        hand_crop, bbox_fallback = detect_hand_fallback(frame, IMG_SIZE)
        if hand_crop is not None:
            bbox = bbox_fallback

    if hand_crop is None:
        return None, 0.0, None

    rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
    preprocessed = preprocess_input(rgb.astype(np.float32))
    batch = np.expand_dims(preprocessed, axis=0)

    probs = model.predict(batch, verbose=0)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    if confidence < confidence_threshold:
        return None, confidence, bbox

    return CLASS_NAMES[pred_idx], confidence, bbox


def run_webcam_demo(model_path=None, camera_id=0, confidence_threshold=0.3,
                    window_size=10):
    model = load_cnn_model(model_path)
    if model is None:
        return

    print("Inicijalizacija MediaPipe HandLandmarker...")
    landmarker = create_hand_landmarker(
        min_detection_confidence=0.3,
        min_presence_confidence=0.3
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"GRESKA: Ne mogu otvoriti kameru (ID={camera_id})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 50)
    print("WEBCAM DEMO - ASL Recognition")
    print("=" * 50)
    print("Pritisnite 'q' za izlaz")
    print("Pritisnite 's' za screenshot")
    print("=" * 50 + "\n")

    prediction_buffer = deque(maxlen=window_size)
    fps_buffer = deque(maxlen=30)
    frame_count = 0

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        label, confidence, bbox = predict_frame(
            model, frame, landmarker, confidence_threshold
        )

        if label is not None:
            prediction_buffer.append(label)

        if bbox is not None:
            x, y, w, h = bbox
            color = (0, 255, 0) if label else (0, 165, 255)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

        if prediction_buffer:
            from collections import Counter
            vote_counts = Counter(prediction_buffer)
            final_label, count = vote_counts.most_common(1)[0]
            vote_confidence = count / len(prediction_buffer)

            text = f"{final_label} ({vote_confidence:.0%})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]

            cv2.rectangle(display_frame, (10, 10), (20 + text_size[0], 20 + text_size[1] + 10),
                          (0, 0, 0), -1)
            cv2.putText(display_frame, text, (15, 15 + text_size[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        if label is not None:
            conf_text = f"Frame: {label} ({confidence:.2f})"
            cv2.putText(display_frame, conf_text, (10, display_frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elapsed = time.time() - start
        fps_buffer.append(1.0 / max(elapsed, 1e-6))
        avg_fps = np.mean(fps_buffer)
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("ASL Recognition", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_path = os.path.join(BASE_DIR, f"screenshot_{frame_count}.png")
            cv2.imwrite(screenshot_path, display_frame)
            print(f"Screenshot sacuvan: {screenshot_path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\nZavrseno. Obradjeno {frame_count} frejmova.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASL Recognition - Webcam Demo")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--window", type=int, default=10,
                        help="Velicina prozora za majority voting")

    args = parser.parse_args()

    run_webcam_demo(
        model_path=args.model,
        camera_id=args.camera,
        confidence_threshold=args.threshold,
        window_size=args.window
    )
