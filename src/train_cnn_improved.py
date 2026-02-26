
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
    GlobalAveragePooling2D, Dropout, Dense, Input, 
    BatchNormalization, RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
SPLIT_JSON = os.path.join(BASE_DIR, "data_split.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CLASS_NAMES = ["cat", "hello", "help", "more", "no",
               "please", "sorry", "thank_you", "what", "yes"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (224, 224)
BATCH_SIZE = 64  
EPOCHS_STAGE1 = 25 
EPOCHS_STAGE2 = 20  
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5
DROPOUT_RATE = 0.5


def load_split_images(split_name):
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    items = split_data[split_name]
    X, y = [], []

    for item in items:
        img_path = os.path.join(FRAMES_DIR, item["path"])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, IMG_SIZE)
        X.append(rgb_resized.astype(np.float32))
        y.append(item["class_idx"])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def create_data_augmentation():
    """Create data augmentation pipeline"""
    return ImageDataGenerator(
        rotation_range=20,           
        width_shift_range=0.2,      
        height_shift_range=0.2,   
        zoom_range=0.15,           
        horizontal_flip=True,      
        brightness_range=[0.8, 1.2], 
        fill_mode='nearest',
        preprocessing_function=preprocess_input 
    )


def create_batch_generator(split_name, batch_size, augment=False, shuffle=True):
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)[split_name]
    
    datagen = create_data_augmentation() if augment else None
    
    while True:
        if shuffle:
            np.random.shuffle(items)
        
        for start_idx in range(0, len(items), batch_size):
            batch_items = items[start_idx:start_idx + batch_size]
            
            X_batch = []
            y_batch = []
            
            for item in batch_items:
                img_path = os.path.join(FRAMES_DIR, item["path"])
                if not os.path.exists(img_path):
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, IMG_SIZE).astype(np.float32)
                
                X_batch.append(rgb_resized)
                y_batch.append(item["class_idx"])
            
            if not X_batch:
                continue
            
            X_batch = np.array(X_batch)
            y_batch = to_categorical(np.array(y_batch), NUM_CLASSES)
            
            if augment and datagen is not None:
                aug_X = []
                for img in X_batch:
                    params = datagen.get_random_transform(img.shape)
                    img_aug = datagen.apply_transform(img, params)
                    img_aug = preprocess_input(img_aug)
                    aug_X.append(img_aug)
                X_batch = np.array(aug_X)
            else:
                X_batch = preprocess_input(X_batch)
            
            yield X_batch, y_batch


def build_model(num_classes, trainable_base=False, unfreeze_from=100):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        pooling="avg"
    )
    
    base_model.trainable = trainable_base
    if trainable_base and unfreeze_from > 0:
        for layer in base_model.layers[:unfreeze_from]:
            layer.trainable = False
    
    inputs = Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=trainable_base)
    
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE / 2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"  Base model: {trainable_count}/{len(base_model.layers)} layers trainable")
    
    return model


def plot_training_history(history_stage1, history_stage2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_stage1 = len(history_stage1.history["loss"])
    epochs_stage2 = len(history_stage2.history["loss"])
    
    acc = history_stage1.history["accuracy"] + history_stage2.history["accuracy"]
    val_acc = history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"]
    loss = history_stage1.history["loss"] + history_stage2.history["loss"]
    val_loss = history_stage1.history["val_loss"] + history_stage2.history["val_loss"]
    
    epochs = range(1, len(acc) + 1)
    
    axes[0].plot(epochs, acc, 'b-', label='Train Accuracy', linewidth=2)
    axes[0].plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
    axes[0].axvline(x=epochs_stage1, color='gray', linestyle='--', label='Fine-tuning starts')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy (2-Stage Training)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, loss, 'b-', label='Train Loss', linewidth=2)
    axes[1].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[1].axvline(x=epochs_stage1, color='gray', linestyle='--', label='Fine-tuning starts')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss (2-Stage Training)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Training history plot saved: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}, annot_kws={'size': 10}
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - Improved CNN', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Confusion matrix plot saved: {save_path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("IMPROVED CNN TRAINING ")
    print("="*70)
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Stage 1 Epochs: {EPOCHS_STAGE1} (frozen base, LR={LEARNING_RATE_STAGE1})")
    print(f"Stage 2 Epochs: {EPOCHS_STAGE2} (fine-tune base, LR={LEARNING_RATE_STAGE2})")
    print("="*70)

    print("\n[1/7] Loading dataset metadata...")
    t0 = time.time()
    
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    
    n_train = len(split_data["train"])
    n_val = len(split_data["val"])
    n_test = len(split_data["test"])
    
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"  Loading time: {time.time()-t0:.1f}s")
    
    y_train = np.array([item["class_idx"] for item in split_data["train"]])
    
    print("  Loading validation set...")
    X_val, y_val = load_split_images("val")
    X_val_pp = preprocess_input(X_val.copy())
    y_val_cat = to_categorical(y_val, NUM_CLASSES)

    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    class_weights = {i: w for i, w in enumerate(cw)}
    print(f"\n  Class weights: {class_weights}")

    print("\n" + "="*70)
    print("STAGE 1: Training classifier with frozen MobileNetV2 base")
    print("="*70)
    
    print("\n[2/7] Creating batch generators...")
    train_generator = create_batch_generator("train", BATCH_SIZE, augment=True, shuffle=True)
    steps_per_epoch = n_train // BATCH_SIZE
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    print("\n[3/7] Building model (frozen base)...")
    model = build_model(NUM_CLASSES, trainable_base=False)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_STAGE1),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    
    stage1_model_path = os.path.join(MODELS_DIR, "cnn_improved_stage1.keras")
    callbacks_stage1 = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=7, 
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            stage1_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,  
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n[4/7] Training Stage 1 (with augmentation)...")
    t0 = time.time()
    
    history_stage1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val_pp, y_val_cat),
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    stage1_time = time.time() - t0
    best_val_acc_s1 = max(history_stage1.history["val_accuracy"])
    print(f"\n  Stage 1 completed: {stage1_time:.1f}s")
    print(f"  Best validation accuracy: {best_val_acc_s1:.4f}")

    print("\n  Loading best Stage 1 model...")
    model = keras.models.load_model(stage1_model_path)

    print("\n" + "="*70)
    print("STAGE 2: Fine-tuning with unfrozen MobileNetV2 layers")
    print("="*70)
    
    print("\n[5/7] Unfreezing base model layers...")
    base_model = model.layers[1]  
    base_model.trainable = True
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"  Unfrozen {trainable_count}/{len(base_model.layers)} layers")
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_STAGE2),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    stage2_model_path = os.path.join(MODELS_DIR, "cnn_improved_final.keras")
    callbacks_stage2 = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=6, 
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            stage2_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,  
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n[6/7] Training Stage 2 (fine-tuning with augmentation)...")
    t0 = time.time()
    
    train_generator_stage2 = create_batch_generator("train", BATCH_SIZE, augment=True, shuffle=True)
    
    history_stage2 = model.fit(
        train_generator_stage2,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val_pp, y_val_cat),
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    stage2_time = time.time() - t0
    best_val_acc_s2 = max(history_stage2.history["val_accuracy"])
    print(f"\n  Stage 2 completed: {stage2_time:.1f}s")
    print(f"  Best validation accuracy: {best_val_acc_s2:.4f}")
    
    print("\n  Loading best Stage 2 model...")
    model = keras.models.load_model(stage2_model_path)

    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    print("\n[7/7] Evaluating on test set...")
    print("  Loading test set...")
    X_test, y_test = load_split_images("test")
    X_test_pp = preprocess_input(X_test.copy())
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    print("  Running evaluation...")
    test_loss, test_acc = model.evaluate(X_test_pp, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test_pp, verbose=0), axis=1)

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")

    report_text = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(f"\nClassification Report:\n{report_text}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    print(f"\nF1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")

    metrics_path = os.path.join(RESULTS_DIR, "cnn_improved_metrics.json")
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "stage1_time": stage1_time,
        "stage2_time": stage2_time,
        "total_time": stage1_time + stage2_time,
        "best_val_acc_stage1": float(best_val_acc_s1),
        "best_val_acc_stage2": float(best_val_acc_s2),
        "classification_report": report_dict
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    history_path = os.path.join(RESULTS_DIR, "cnn_improved_history.json")
    history_combined = {
        "stage1": {
            "accuracy": [float(x) for x in history_stage1.history["accuracy"]],
            "val_accuracy": [float(x) for x in history_stage1.history["val_accuracy"]],
            "loss": [float(x) for x in history_stage1.history["loss"]],
            "val_loss": [float(x) for x in history_stage1.history["val_loss"]],
        },
        "stage2": {
            "accuracy": [float(x) for x in history_stage2.history["accuracy"]],
            "val_accuracy": [float(x) for x in history_stage2.history["val_accuracy"]],
            "loss": [float(x) for x in history_stage2.history["loss"]],
            "val_loss": [float(x) for x in history_stage2.history["val_loss"]],
        }
    }
    with open(history_path, "w") as f:
        json.dump(history_combined, f, indent=2)
    print(f"  History saved: {history_path}")

    print("\n  Generating visualizations...")
    history_plot_path = os.path.join(RESULTS_DIR, "cnn_improved_history.png")
    plot_training_history(history_stage1, history_stage2, history_plot_path)

    cm_plot_path = os.path.join(RESULTS_DIR, "cnn_improved_confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_plot_path)

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"F1 (macro):    {f1_macro:.4f}")
    print(f"Total Time:    {stage1_time + stage2_time:.1f}s")
    print(f"Final Model:   {stage2_model_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
