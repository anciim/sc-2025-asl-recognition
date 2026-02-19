import os
import sys
import json
import numpy as np
import cv2
import time

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
SPLIT_JSON = os.path.join(BASE_DIR, "data_split.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = ["cat", "hello", "help", "more", "no",
               "please", "sorry", "thank_you", "what", "yes"]
NUM_CLASSES = len(CLASS_NAMES)


def load_split_data(frames_dir, split_json_path, split_name):
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    items = split_data[split_name]
    X, y, videos = [], [], []

    for item in items:
        img_path = os.path.join(frames_dir, item["path"])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_input(rgb.astype(np.float32))
        X.append(preprocessed)
        y.append(item["class_idx"])
        videos.append(item["video"])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), videos


def evaluate_per_frame(model, X_test, y_test):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    return {
        "accuracy": acc,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
        "precision_macro": prec,
        "recall_macro": rec,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs
    }


def evaluate_per_video(model, X_test, y_test, videos):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_frame = np.argmax(y_pred_probs, axis=1)

    from collections import defaultdict
    video_data = defaultdict(lambda: {"preds": [], "probs": [], "label": None})

    for i in range(len(y_test)):
        video_key = videos[i]
        video_data[video_key]["preds"].append(y_pred_frame[i])
        video_data[video_key]["probs"].append(y_pred_probs[i])
        video_data[video_key]["label"] = y_test[i]

    video_true, video_pred = [], []

    for video_key, data in video_data.items():
        true_label = data["label"]

        preds = np.array(data["preds"])
        counts = np.bincount(preds, minlength=NUM_CLASSES)
        majority_pred = np.argmax(counts)

        avg_probs = np.mean(data["probs"], axis=0)
        avg_pred = np.argmax(avg_probs)

        video_true.append(true_label)
        video_pred.append(majority_pred)

    video_true = np.array(video_true)
    video_pred = np.array(video_pred)

    acc = accuracy_score(video_true, video_pred)
    f1_m = f1_score(video_true, video_pred, average="macro")
    cm = confusion_matrix(video_true, video_pred)
    report = classification_report(video_true, video_pred, target_names=CLASS_NAMES)

    return {
        "accuracy": acc,
        "f1_macro": f1_m,
        "confusion_matrix": cm,
        "report": report,
        "num_videos": len(video_data),
        "video_true": video_true,
        "video_pred": video_pred
    }


def measure_inference_speed(model, X_test, n_runs=3):
    single_img = X_test[:1]

    _ = model.predict(single_img, verbose=0)

    times_single = []
    for _ in range(n_runs * 10):
        start = time.time()
        model.predict(single_img, verbose=0)
        times_single.append(time.time() - start)

    times_batch = []
    for _ in range(n_runs):
        start = time.time()
        model.predict(X_test, verbose=0)
        times_batch.append(time.time() - start)

    avg_single = np.mean(times_single) * 1000
    avg_per_image = np.mean(times_batch) / len(X_test) * 1000
    fps = 1000.0 / avg_per_image if avg_per_image > 0 else 0

    return {
        "avg_single_ms": float(avg_single),
        "avg_per_image_batch_ms": float(avg_per_image),
        "estimated_fps": float(fps),
        "batch_size_tested": len(X_test)
    }


def plot_confusion_matrix(cm, class_names, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Stvarna klasa",
        xlabel="Predvidjena klasa",
        title="Confusion Matrix"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(report_dict, class_names, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precisions, recalls, f1s = [], [], []
    for cls in class_names:
        if cls in report_dict:
            precisions.append(report_dict[cls]["precision"])
            recalls.append(report_dict[cls]["recall"])
            f1s.append(report_dict[cls]["f1-score"])
        else:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="#2196F3")
    ax.bar(x, recalls, width, label="Recall", color="#4CAF50")
    ax.bar(x + width, f1s, width, label="F1-Score", color="#FF9800")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrike")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_model(model_path=None):
    print("=" * 60)
    print("EVALUACIJA CNN MODELA")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

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
            print("GRESKA: Nije pronadjen nijedan model!")
            print(f"  Trazeno u: {MODELS_DIR}")
            return

    print(f"\n  Model: {model_path}")

    print("\n[1/5] Ucitavanje modela...")
    model = load_model(model_path)

    print("\n[2/5] Ucitavanje test podataka...")
    X_test, y_test, videos = load_split_data(FRAMES_DIR, SPLIT_JSON, "test")
    print(f"  Test set: {len(X_test)} slika iz {len(set(videos))} videa")

    print("\n[3/5] Evaluacija po frejmu...")
    frame_results = evaluate_per_frame(model, X_test, y_test)
    print(f"\n  Frame-level Accuracy: {frame_results['accuracy']:.4f}")
    print(f"  Frame-level F1 (macro): {frame_results['f1_macro']:.4f}")
    print(f"  Frame-level F1 (weighted): {frame_results['f1_weighted']:.4f}")
    print(f"  Precision (macro): {frame_results['precision_macro']:.4f}")
    print(f"  Recall (macro): {frame_results['recall_macro']:.4f}")
    print(f"\n{frame_results['report']}")

    print("\n[4/5] Evaluacija po videu (majority voting)...")
    video_results = evaluate_per_video(model, X_test, y_test, videos)
    print(f"\n  Video-level Accuracy: {video_results['accuracy']:.4f}")
    print(f"  Video-level F1 (macro): {video_results['f1_macro']:.4f}")
    print(f"  Broj test videa: {video_results['num_videos']}")
    print(f"\n{video_results['report']}")

    print("\n[5/5] Merenje brzine inferencije...")
    speed = measure_inference_speed(model, X_test)
    print(f"  Prosecno vreme (1 slika): {speed['avg_single_ms']:.1f} ms")
    print(f"  Prosecno vreme (batch): {speed['avg_per_image_batch_ms']:.1f} ms/slika")
    print(f"  Estimirani FPS: {speed['estimated_fps']:.1f}")

    print("\nCuvanje rezultata...")

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(frame_results["confusion_matrix"], CLASS_NAMES, cm_path)
    print(f"  Confusion matrix: {cm_path}")

    cm_video_path = os.path.join(RESULTS_DIR, "confusion_matrix_video.png")
    plot_confusion_matrix(video_results["confusion_matrix"], CLASS_NAMES, cm_video_path)
    print(f"  Confusion matrix (video): {cm_video_path}")

    report_dict = classification_report(
        y_test, frame_results["y_pred"],
        target_names=CLASS_NAMES, output_dict=True
    )
    metrics_plot_path = os.path.join(RESULTS_DIR, "per_class_metrics.png")
    plot_per_class_metrics(report_dict, CLASS_NAMES, metrics_plot_path)
    print(f"  Per-class metrike: {metrics_plot_path}")

    eval_results = {
        "model_path": model_path,
        "frame_level": {
            "accuracy": float(frame_results["accuracy"]),
            "f1_macro": float(frame_results["f1_macro"]),
            "f1_weighted": float(frame_results["f1_weighted"]),
            "precision_macro": float(frame_results["precision_macro"]),
            "recall_macro": float(frame_results["recall_macro"]),
            "confusion_matrix": frame_results["confusion_matrix"].tolist()
        },
        "video_level": {
            "accuracy": float(video_results["accuracy"]),
            "f1_macro": float(video_results["f1_macro"]),
            "num_videos": video_results["num_videos"],
            "confusion_matrix": video_results["confusion_matrix"].tolist()
        },
        "inference_speed": speed,
        "per_class": {}
    }

    for cls in CLASS_NAMES:
        if cls in report_dict:
            eval_results["per_class"][cls] = {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"]
            }

    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"  Kompletni rezultati: {eval_path}")

    print("\n" + "=" * 60)
    print("REZIME EVALUACIJE")
    print("=" * 60)
    print(f"  Frame-level Accuracy:  {frame_results['accuracy']:.4f}")
    print(f"  Video-level Accuracy:  {video_results['accuracy']:.4f}")
    print(f"  F1 Score (macro):      {frame_results['f1_macro']:.4f}")
    print(f"  Inference Speed:       {speed['estimated_fps']:.1f} FPS")
    print("=" * 60)

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASL CNN Evaluacija")
    parser.add_argument("--model", type=str, default=None,
                        help="Putanja do modela (.h5)")

    args = parser.parse_args()
    evaluate_model(args.model)
