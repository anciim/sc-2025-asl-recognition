import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import create_hand_landmarker, scale_to_range

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = ["cat", "hello", "help", "more", "no",
               "please", "sorry", "thank_you", "what", "yes"]

CONFIDENCE_THRESHOLD = 0.4
PREDICTION_BUFFER_SIZE = 10 
IMG_SIZE = (224, 224)
FONT = cv2.FONT_HERSHEY_SIMPLEX
USE_BOTH_HANDS = True  


class RealtimeASLRecognizer:
    def __init__(self, model_path):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded: {model_path}")
        
        print("Initializing MediaPipe hand detector...")
        self.hand_landmarker = create_hand_landmarker(
            min_detection_confidence=0.3,
            min_presence_confidence=0.3
        )
        print("Hand detector ready!")
        
        self.prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
        self.class_names = CLASS_NAMES
        
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
    
    def process_hand_region(self, frame, detection_result, use_both_hands=False):
        if not detection_result.hand_landmarks:
            return None
        
        h, w, _ = frame.shape
        
        if use_both_hands and len(detection_result.hand_landmarks) > 1:
            all_x_coords = []
            all_y_coords = []
            
            for hand_landmarks in detection_result.hand_landmarks:
                all_x_coords.extend([lm.x * w for lm in hand_landmarks])
                all_y_coords.extend([lm.y * h for lm in hand_landmarks])
            
            x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
            y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
            num_hands = len(detection_result.hand_landmarks)
        else:
            hand_landmarks = detection_result.hand_landmarks[0]
            x_coords = [lm.x * w for lm in hand_landmarks]
            y_coords = [lm.y * h for lm in hand_landmarks]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            num_hands = 1
        
        margin = 30
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        hand_crop = frame[y_min:y_max, x_min:x_max]
        
        if hand_crop.size == 0:
            return None
        
        hand_resized = cv2.resize(hand_crop, IMG_SIZE)
        
        return hand_resized, (x_min, y_min, x_max, y_max), num_hands
    
    def predict(self, hand_image):
        rgb = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
        scaled = scale_to_range(rgb)
        input_frame = np.expand_dims(scaled, axis=0)
        
        pred_probs = self.model.predict(input_frame, verbose=0)[0]
        pred_class = np.argmax(pred_probs)
        confidence = pred_probs[pred_class]
        
        return pred_class, confidence, pred_probs
    
    def get_smoothed_prediction(self):
        if not self.prediction_buffer:
            return None, 0.0
        
        votes = Counter(self.prediction_buffer)
        most_common_class, count = votes.most_common(1)[0]
        
        confidence = count / len(self.prediction_buffer)
        
        return most_common_class, confidence
    
    def draw_prediction_panel(self, frame, pred_class, confidence, all_probs, smoothed_class, smoothed_conf):
        """Draw prediction information on frame"""
        h, w, _ = frame.shape
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        if pred_class is not None:
            text = f"Current: {self.class_names[pred_class].upper()}"
            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, text, (20, 40), FONT, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 70), FONT, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No hand detected", (20, 40), FONT, 0.9, (0, 0, 255), 2)
        
        if smoothed_class is not None:
            text = f"Predicted: {self.class_names[smoothed_class].upper()}"
            cv2.putText(frame, text, (20, 110), FONT, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, f"Stability: {smoothed_conf:.2f}", (20, 145), FONT, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 180), FONT, 0.5, (200, 200, 200), 1)
        
        if all_probs is not None:
            self.draw_top_predictions(frame, all_probs)
    
    def draw_top_predictions(self, frame, probs):
        h, w, _ = frame.shape
        
        top3_indices = np.argsort(probs)[-3:][::-1]
        
        bar_width = 300
        bar_height = 25
        x_start = w - bar_width - 20
        y_start = 20
        
        for i, idx in enumerate(top3_indices):
            y = y_start + i * (bar_height + 10)
            conf = probs[idx]
            
            cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height), (50, 50, 50), -1)
            
            bar_len = int(bar_width * conf)
            color = (0, 255, 0) if i == 0 else (100, 200, 255)
            cv2.rectangle(frame, (x_start, y), (x_start + bar_len, y + bar_height), color, -1)
            
            text = f"{self.class_names[idx]}: {conf:.2f}"
            cv2.putText(frame, text, (x_start + 5, y + 18), FONT, 0.5, (255, 255, 255), 1)
    
    def draw_hand_box(self, frame, bbox, num_hands=1):
        x_min, y_min, x_max, y_max = bbox
        color = (0, 255, 255) if num_hands > 1 else (0, 255, 0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        label = f"{num_hands} Hand" + ("s" if num_hands > 1 else "")
        cv2.putText(frame, label, (x_min, y_min - 10), FONT, 0.6, color, 2)
    
    def update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
    
    def reset_predictions(self):
        self.prediction_buffer.clear()
        print("Predictions reset!")
    
    def run(self):
        print("\n" + "="*70)
        print("REAL-TIME ASL RECOGNITION")
        print("="*70)
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset predictions")
        print("  Space - Pause/Resume")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot open webcam!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        paused = False
        
        print("Webcam opened successfully!")
        print("Starting recognition...\n")
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("Failed to grab frame")
                        break
                    
                    frame = cv2.flip(frame, 1)
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    import mediapipe as mp
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    detection_result = self.hand_landmarker.detect(mp_image)
                    
                    pred_class = None
                    confidence = 0.0
                    all_probs = None
                    bbox = None
                    num_hands = 0
                    
                    if detection_result.hand_landmarks:
                        result = self.process_hand_region(frame, detection_result, 
                                                         use_both_hands=USE_BOTH_HANDS)
                        
                        if result is not None:
                            hand_image, bbox, num_hands = result
                            
                            pred_class, confidence, all_probs = self.predict(hand_image)
                            
                            if confidence > CONFIDENCE_THRESHOLD:
                                self.prediction_buffer.append(pred_class)
                            
                            self.draw_hand_box(frame, bbox, num_hands)
                    
                    smoothed_class, smoothed_conf = self.get_smoothed_prediction()
                    
                    self.draw_prediction_panel(frame, pred_class, confidence, all_probs, smoothed_class, smoothed_conf)
                    
                    self.update_fps()
                
                cv2.imshow('ASL Real-Time Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    self.reset_predictions()
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("\nWebcam closed")


def main():
    model_path = os.path.join(MODELS_DIR, "cnn_improved_final.keras")
    if not os.path.exists(model_path):
        print("Improved model not found, trying original CNN...")
        model_path = os.path.join(MODELS_DIR, "cnn_model_final.keras")
    
    if not os.path.exists(model_path):
        print("ERROR: No trained model found!")
        print("Please train a model first using train_cnn.py or train_cnn_improved.py")
        return
    
    recognizer = RealtimeASLRecognizer(model_path)
    
    recognizer.run()

if __name__ == "__main__":
    main()
