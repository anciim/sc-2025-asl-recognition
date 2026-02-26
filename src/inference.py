import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    extract_frames, 
    process_frame_with_hand_detection,
    create_hand_landmarker,
    scale_to_range,
    CLASS_NAMES
)

CONFIDENCE_THRESHOLD = 0.3
SEQ_LENGTH = 15


class ASLRecognizer:
    
    def __init__(self, cnn_model_path=None, lstm_model_path=None):
        self.class_names = CLASS_NAMES
        
        if cnn_model_path and os.path.exists(cnn_model_path):
            print(f"Učitavam CNN model: {cnn_model_path}")
            self.cnn_model = tf.keras.models.load_model(cnn_model_path)
        else:
            self.cnn_model = None
            print("CNN model nije učitan")
        
        # Učitaj LSTM model (video sekvenca)
        if lstm_model_path and os.path.exists(lstm_model_path):
            print(f"Učitavam LSTM model: {lstm_model_path}")
            self.lstm_model = tf.keras.models.load_model(lstm_model_path)
        else:
            self.lstm_model = None
            print("LSTM model nije učitan")
        
        self.hand_landmarker = create_hand_landmarker(
            min_detection_confidence=0.3,
            min_presence_confidence=0.3
        )
    
    def predict_frame_by_frame(self, video_path):
        if self.cnn_model is None:
            return None, 0.0, []
        
        frames = extract_frames(video_path, target_fps=10, max_frames=30)
        
        if not frames:
            return None, 0.0, []
        
        predictions = []
        confidences = []
        
        for frame in frames:
            hand_crop, _, _ = process_frame_with_hand_detection(
                frame, self.hand_landmarker, target_size=(224, 224)
            )
            
            if hand_crop is None:
                continue
            
            rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            scaled = scale_to_range(rgb)
            input_frame = np.expand_dims(scaled, axis=0)
            
            pred_probs = self.cnn_model.predict(input_frame, verbose=0)[0]
            pred_class = np.argmax(pred_probs)
            confidence = pred_probs[pred_class]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                predictions.append(pred_class)
                confidences.append(confidence)
        
        if not predictions:
            return None, 0.0, []
        
        vote_counts = Counter(predictions)
        most_common_class, vote_count = vote_counts.most_common(1)[0]
        
        class_confidences = [
            conf for pred, conf in zip(predictions, confidences) 
            if pred == most_common_class
        ]
        avg_confidence = np.mean(class_confidences)
        
        return most_common_class, avg_confidence, predictions
    
    def predict_video_sequence(self, video_path):
        if self.lstm_model is None:
            return None, 0.0
        
        frames = extract_frames(video_path, target_fps=10, max_frames=30)
        
        if not frames:
            return None, 0.0
        
        processed_frames = []
        for frame in frames:
            hand_crop, _, _ = process_frame_with_hand_detection(
                frame, self.hand_landmarker, target_size=(224, 224)
            )
            
            if hand_crop is not None:
                rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                scaled = scale_to_range(rgb)
                processed_frames.append(scaled)
        
        if not processed_frames:
            return None, 0.0
        
        if len(processed_frames) >= SEQ_LENGTH:
            indices = np.linspace(0, len(processed_frames) - 1, SEQ_LENGTH, dtype=int)
            sequence = [processed_frames[i] for i in indices]
        else:
            sequence = processed_frames + [processed_frames[-1]] * (SEQ_LENGTH - len(processed_frames))
        
        sequence_array = np.array([sequence]) 
        
        pred_probs = self.lstm_model.predict(sequence_array, verbose=0)[0]
        pred_class = np.argmax(pred_probs)
        confidence = pred_probs[pred_class]
        
        return pred_class, confidence
    
    def predict(self, video_path, method='both'):
        results = {}
        
        if method in ['cnn', 'both'] and self.cnn_model is not None:
            print("\n[CNN] Frame-by-frame + Majority Voting...")
            pred_class, confidence, all_preds = self.predict_frame_by_frame(video_path)
            
            if pred_class is not None:
                results['cnn'] = {
                    'class': self.class_names[pred_class],
                    'class_idx': int(pred_class),
                    'confidence': float(confidence),
                    'num_frames': len(all_preds)
                }
                print(f"  Rezultat: {self.class_names[pred_class]} (confidence: {confidence:.3f})")
            else:
                results['cnn'] = None
                print("  Nema validinih predikcija")
        
        if method in ['lstm', 'both'] and self.lstm_model is not None:
            print("\n[LSTM] Video sekvenca...")
            pred_class, confidence = self.predict_video_sequence(video_path)
            
            if pred_class is not None:
                results['lstm'] = {
                    'class': self.class_names[pred_class],
                    'class_idx': int(pred_class),
                    'confidence': float(confidence)
                }
                print(f"  Rezultat: {self.class_names[pred_class]} (confidence: {confidence:.3f})")
            else:
                results['lstm'] = None
                print("  Nema validinih predikcija")
        
        return results
    
    def __del__(self):
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    cnn_model_path = os.path.join(models_dir, "cnn_model_final.keras")
    lstm_model_path = os.path.join(models_dir, "lstm_model.keras")
    
    recognizer = ASLRecognizer(
        cnn_model_path=cnn_model_path,
        lstm_model_path=lstm_model_path
    )
    
    test_video = os.path.join(base_dir, "data", "hello", "27172.mp4")
    
    if os.path.exists(test_video):
        print(f"\nTest video: {test_video}")
        print("=" * 60)
        
        results = recognizer.predict(test_video, method='both')
        
        print("\n" + "=" * 60)
        print("REZULTATI:")
        print("=" * 60)
        
        if 'cnn' in results and results['cnn']:
            print(f"CNN:  {results['cnn']['class']} ({results['cnn']['confidence']:.2%})")
        
        if 'lstm' in results and results['lstm']:
            print(f"LSTM: {results['lstm']['class']} ({results['lstm']['confidence']:.2%})")
        
        print("=" * 60)
    else:
        print(f"Test video ne postoji: {test_video}")


if __name__ == "__main__":
    main()
