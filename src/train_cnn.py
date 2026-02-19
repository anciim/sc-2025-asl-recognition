import os
import sys
import json
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, GaussianNoise
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
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
        X.append(rgb.astype(np.float32))
        y.append(item["class_idx"])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def extract_features_batched(images, base_model, batch_size=32):
    all_features = []
    n = len(images)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = preprocess_input(images[start:end].copy())
        feats = base_model.predict(batch, verbose=0)
        all_features.append(feats)
        if (start // batch_size) % 5 == 0:
            print(f"    {end}/{n}")
    return np.concatenate(all_features, axis=0)


def build_classifier(input_dim, num_classes):
    return Sequential([
        Input(shape=(input_dim,)),
        GaussianNoise(0.1),
        Dropout(0.5),
        Dense(128, activation="relu", kernel_regularizer=l2(1e-3)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])


def build_full_model(base_model, classifier):
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    outputs = classifier(x)
    return Model(inputs, outputs)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("CNN TRENING - MobileNetV2 Feature Extraction (BRZI)")
    print("=" * 60)

    print("\n[1/6] Ucitavanje slika...")
    t0 = time.time()
    X_train, y_train = load_split_images("train")
    X_val, y_val = load_split_images("val")
    X_test, y_test = load_split_images("test")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} ({time.time()-t0:.1f}s)")

    print("\n[2/6] Ekstrakcija MobileNetV2 feature-a (JEDNOM, batch-by-batch)...")
    t0 = time.time()
    base_model = MobileNetV2(
        weights="imagenet", include_top=False,
        input_shape=(224, 224, 3), pooling="avg"
    )
    base_model.trainable = False

    print("  Train features...")
    feat_train = extract_features_batched(X_train, base_model, batch_size=16)
    print("  Val features...")
    feat_val = extract_features_batched(X_val, base_model, batch_size=16)
    print("  Test features...")
    feat_test = extract_features_batched(X_test, base_model, batch_size=16)
    feat_dim = feat_train.shape[1]
    print(f"  Dimenzija: {feat_dim}, Vreme: {time.time()-t0:.1f}s")

    del X_train

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    class_weights = {i: w for i, w in enumerate(cw)}

    print("\n[3/6] Trening klasifikatora na feature vektorima...")
    classifier = build_classifier(feat_dim, NUM_CLASSES)
    classifier.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    classifier.summary()

    clf_path = os.path.join(MODELS_DIR, "classifier_head.keras")
    callbacks_clf = [
        EarlyStopping(monitor="val_accuracy", patience=30, mode="max", restore_best_weights=True, verbose=1),
        ModelCheckpoint(clf_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]

    t0 = time.time()
    history_clf = classifier.fit(
        feat_train, y_train_cat,
        validation_data=(feat_val, y_val_cat),
        epochs=300, batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks_clf, verbose=1
    )
    clf_time = time.time() - t0
    best_val = max(history_clf.history["val_accuracy"])
    print(f"\n  Klasifikator: {clf_time:.1f}s, best val acc: {best_val:.4f}")

    print("\n[4/6] Sklapanje kompletnog modela...")
    classifier = keras.models.load_model(clf_path)
    full_model = build_full_model(base_model, classifier)

    print("\n[5/6] Fine-tuning poslednjih slojeva...")
    base_model.trainable = True
    for layer in base_model.layers[:130]:
        layer.trainable = False

    trainable_n = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Odmrznuto {trainable_n}/{len(base_model.layers)} slojeva baze")

    full_model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    X_val_pp = preprocess_input(X_val.copy())

    ft_path = os.path.join(MODELS_DIR, "cnn_model_finetuned.keras")
    callbacks_ft = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ft_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    def ft_data_generator(split="train", batch_size=16):
        with open(SPLIT_JSON, "r", encoding="utf-8") as f:
            items = json.load(f)[split]
        sample_weights_map = {}
        for item in items:
            ci = item["class_idx"]
            if ci in class_weights:
                sample_weights_map[id(item)] = class_weights[ci]
        indices = np.arange(len(items))
        while True:
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start+batch_size]
                imgs, labs, sw = [], [], []
                for i in batch_idx:
                    p = os.path.join(FRAMES_DIR, items[i]["path"])
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                    imgs.append(rgb)
                    ci = items[i]["class_idx"]
                    labs.append(ci)
                    sw.append(class_weights.get(ci, 1.0))
                if not imgs:
                    continue
                X_b = preprocess_input(np.array(imgs))
                y_b = to_categorical(np.array(labs), NUM_CLASSES)
                yield X_b, y_b, np.array(sw, dtype=np.float32)

    n_train = len(y_train)
    ft_bs = 16
    steps_per_epoch = n_train // ft_bs

    t0 = time.time()
    history_ft = full_model.fit(
        ft_data_generator("train", ft_bs),
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val_pp, y_val_cat),
        epochs=15,
        callbacks=callbacks_ft, verbose=1
    )
    ft_time = time.time() - t0
    best_ft = max(history_ft.history["val_accuracy"])
    print(f"\n  Fine-tune: {ft_time:.1f}s, best val acc: {best_ft:.4f}")

    print("\n[6/6] Evaluacija na test setu...")
    X_test_pp = preprocess_input(X_test.copy())
    test_loss, test_acc = full_model.evaluate(X_test_pp, y_test_cat, verbose=0)
    y_pred = np.argmax(full_model.predict(X_test_pp, verbose=0), axis=1)

    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")

    report_text = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")
    print(f"\n{report_text}")
    print(f"Confusion Matrix:\n{cm}")

    hist_all = {
        "accuracy": history_clf.history["accuracy"] + history_ft.history["accuracy"],
        "val_accuracy": history_clf.history["val_accuracy"] + history_ft.history["val_accuracy"],
        "loss": history_clf.history["loss"] + history_ft.history["loss"],
        "val_loss": history_clf.history["val_loss"] + history_ft.history["val_loss"],
    }

    save_plots(hist_all)

    final_path = os.path.join(MODELS_DIR, "cnn_model_final.keras")
    full_model.save(final_path)
    print(f"\n  Model sacuvan: {final_path}")

    total_time = clf_time + ft_time
    best_val_all = max(best_val, best_ft) if history_ft.history["val_accuracy"] else best_val
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "f1_macro": float(f1_m),
        "f1_weighted": float(f1_w),
        "training_time_seconds": float(total_time),
        "best_val_accuracy": float(best_val_all),
        "total_epochs": len(hist_all["loss"]),
        "class_names": CLASS_NAMES,
        "per_class_metrics": {},
        "confusion_matrix": cm.tolist()
    }
    for cls in CLASS_NAMES:
        if cls in report_dict:
            metrics["per_class_metrics"][cls] = {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1-score": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"]
            }

    with open(os.path.join(RESULTS_DIR, "cnn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, "cnn_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist_all.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("TRENING ZAVRSEN")
    print("=" * 60)
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  F1 (macro):    {f1_m:.4f}")
    print(f"  Vreme: klasifikator {clf_time:.0f}s + fine-tune {ft_time:.0f}s = {total_time:.0f}s")


def save_plots(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["accuracy"]) + 1)
    axes[0].plot(epochs, history["accuracy"], "b-", label="Train")
    axes[0].plot(epochs, history["val_accuracy"], "r-", label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoha")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["loss"], "b-", label="Train")
    axes[1].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoha")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_training_curves.png"), dpi=150)
    plt.close()
    print("  Grafici sacuvani.")


if __name__ == "__main__":
    main()
