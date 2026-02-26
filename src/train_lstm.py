
import os
import sys
import json
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, TimeDistributed, Input,
    GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
SPLIT_JSON = os.path.join(BASE_DIR, "data_split.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = ["cat", "hello", "help", "more", "no",
               "please", "sorry", "thank_you", "what", "yes"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (224, 224)
SEQUENCE_LENGTH = 10  
BATCH_SIZE = 16  
EPOCHS = 50
LEARNING_RATE = 1e-4
LSTM_UNITS = 128


def load_video_sequences(split="train"):
    """
    Load video sequences by grouping frames from the same video
    Returns sequences of fixed length (SEQUENCE_LENGTH frames)
    """
    print(f"\nLoading {split} sequences...")
    
    with open(SPLIT_JSON, "r") as f:
        data_split = json.load(f)
    
    frame_paths = data_split[split]
    print(f"  Found {len(frame_paths)} frames")
    
    video_groups = {}
    for frame_path in frame_paths:
        parts = frame_path.split("/")
        class_name = parts[1]
        video_name = parts[2]
        video_key = f"{class_name}/{video_name}"
        
        if video_key not in video_groups:
            video_groups[video_key] = []
        video_groups[video_key].append(frame_path)
    
    print(f"  Grouped into {len(video_groups)} videos")
    
    sequences = []
    labels = []
    
    for video_key, frames in video_groups.items():
        class_name = video_key.split("/")[0]
        class_idx = CLASS_NAMES.index(class_name)
        
        frames = sorted(frames)
        
        if len(frames) >= SEQUENCE_LENGTH:
            indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
            selected_frames = [frames[i] for i in indices]
        else:
            selected_frames = frames + [frames[-1]] * (SEQUENCE_LENGTH - len(frames))
        
        sequence = []
        for frame_path in selected_frames:
            full_path = os.path.join(BASE_DIR, frame_path)
            img = cv2.imread(full_path)
            if img is None:
                print(f"  Warning: Could not load {frame_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            sequence.append(img)
        
        if len(sequence) == SEQUENCE_LENGTH:
            sequences.append(sequence)
            labels.append(class_idx)
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"  Created {len(sequences)} sequences of shape {sequences.shape}")
    print(f"  Label distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = np.sum(labels == i)
        print(f"    {class_name:12s}: {count:3d} videos")
    
    return sequences, labels


def build_cnn_lstm_model(num_classes, lstm_units=128):
    print("\nBuilding CNN+LSTM model...")
    
    input_shape = (SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    inputs = Input(shape=input_shape, name="video_input")
    
    base_cnn = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_cnn.trainable = False  
    
    cnn_model = Sequential([
        base_cnn,
        GlobalAveragePooling2D()
    ])
    
    features = TimeDistributed(cnn_model, name="cnn_features")(inputs)
    
    lstm_out = LSTM(lstm_units, return_sequences=False, name="lstm_temporal")(features)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.5)(lstm_out)
    
    x = Dense(256, activation="relu", name="dense_1")(lstm_out)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation="softmax", name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_ASL")
    
    print(f"  Model architecture:")
    print(f"    Input: {input_shape}")
    print(f"    CNN: MobileNetV2 (frozen)")
    print(f"    LSTM units: {lstm_units}")
    print(f"    Output classes: {num_classes}")
    
    return model


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val", linewidth=2)
    axes[0].set_title("Model Accuracy (CNN+LSTM)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history["loss"], label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val", linewidth=2)
    axes[1].set_title("Model Loss (CNN+LSTM)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved training history to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Count"}
    )
    plt.title("Confusion Matrix (CNN+LSTM)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved confusion matrix to {save_path}")


def main():
    print("="*70)
    print("CNN+LSTM MODEL TRAINING - ASL Recognition")
    print("="*70)
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"Image size: {IMG_SIZE}")
    print(f"LSTM units: {LSTM_UNITS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    results_dir = os.path.join(BASE_DIR, "results_lstm")
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("LOADING VIDEO SEQUENCES")
    print("="*70)
    
    X_train, y_train = load_video_sequences("train")
    X_val, y_val = load_video_sequences("val")
    X_test, y_test = load_video_sequences("test")
    
    print(f"\nDataset summary:")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Val:   {len(X_val)} sequences")
    print(f"  Test:  {len(X_test)} sequences")
    
    print("\nPreprocessing images...")
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    X_test = preprocess_input(X_test)
    
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    class_weights = {i: w for i, w in enumerate(cw)}
    print(f"\nClass weights: {class_weights}")
    
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    model = build_cnn_lstm_model(NUM_CLASSES, lstm_units=LSTM_UNITS)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\nStarting training...")
    t0 = time.time()
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    training_time = time.time() - t0
    print(f"\nTraining completed in {training_time:.1f}s ({training_time/60:.1f}m)")
    
    print("\nLoading best model...")
    model = keras.models.load_model(model_path)
    
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\n  F1 Score (macro): {f1_macro:.4f}")
    print(f"  F1 Score (weighted): {f1_weighted:.4f}")
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    history_path = os.path.join(results_dir, "lstm_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"  Saved history to {history_path}")
    
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "training_time_seconds": float(training_time),
        "sequence_length": SEQUENCE_LENGTH,
        "lstm_units": LSTM_UNITS,
        "num_train_sequences": len(X_train),
        "num_val_sequences": len(X_val),
        "num_test_sequences": len(X_test)
    }
    metrics_path = os.path.join(results_dir, "lstm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")
    
    history_plot_path = os.path.join(results_dir, "lstm_training_history.png")
    plot_training_history(history, history_plot_path)
    
    cm_path = os.path.join(results_dir, "lstm_confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, cm_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"F1 score (macro): {f1_macro:.4f}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print("="*70)


if __name__ == "__main__":
    main()
