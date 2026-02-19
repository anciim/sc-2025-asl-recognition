import cv2
import numpy as np
import os
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_detection import (
    create_hand_landmarker,
    detect_hand,
    detect_hand_fallback,
    extract_hand_landmarks_as_features,
    image_gray,
    image_bin,
    resize_region,
    dilate,
    erode,
    load_image
)

CLASS_NAMES = ["cat", "hello", "help", "more", "no",
               "please", "sorry", "thank_you", "what", "yes"]

TARGET_SIZE = (224, 224)
EXTRACT_FPS = 10


def extract_frames(video_path, output_dir=None, target_fps=EXTRACT_FPS, max_frames=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"GREŠKA: Ne mogu da otvorim video: {video_path}")
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps <= 0:
        original_fps = 30.0

    frame_interval = max(1, int(original_fps / target_fps))

    frames = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            frames.append(frame)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                frame_path = os.path.join(output_dir, f"frame_{frame_num:04d}.jpg")
                cv2.imwrite(frame_path, frame)

            if len(frames) >= max_frames:
                break

        frame_num += 1

    cap.release()
    return frames


def process_frame_with_hand_detection(frame, landmarker, target_size=TARGET_SIZE):
    hand_crop, landmarks, bbox = detect_hand(frame, landmarker, target_size)

    if hand_crop is not None:
        return hand_crop, landmarks, bbox

    hand_crop, bbox = detect_hand_fallback(frame, target_size)
    if hand_crop is not None:
        return hand_crop, None, bbox

    return None, None, None


def process_video_frames(video_path, landmarker, target_fps=EXTRACT_FPS,
                         target_size=TARGET_SIZE, max_frames=30):
    raw_frames = extract_frames(video_path, target_fps=target_fps, max_frames=max_frames)

    if not raw_frames:
        return [], [], {"video": video_path, "total_frames": 0, "detected": 0}

    hand_frames = []
    all_landmarks = []
    detected_count = 0

    for frame in raw_frames:
        hand_crop, landmarks, bbox = process_frame_with_hand_detection(
            frame, landmarker, target_size
        )

        if hand_crop is not None:
            hand_frames.append(hand_crop)
            detected_count += 1

            if landmarks is not None:
                all_landmarks.append(extract_hand_landmarks_as_features(landmarks))
            else:
                all_landmarks.append(None)

    metadata = {
        "video": os.path.basename(video_path),
        "total_frames": len(raw_frames),
        "detected": detected_count,
        "detection_rate": detected_count / max(1, len(raw_frames))
    }

    return hand_frames, all_landmarks, metadata


def scale_to_range(image):
    return image / 255.0


def prepare_for_model(frames):
    prepared = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scaled = scale_to_range(rgb)
        prepared.append(scaled)
    return np.array(prepared, dtype=np.float32)


def preprocess_dataset(data_dir, output_dir, target_fps=EXTRACT_FPS,
                       target_size=TARGET_SIZE, max_frames=30):
    print("=" * 60)
    print("PREPROCESSING DATASETA")
    print("=" * 60)
    print(f"Ulaz: {data_dir}")
    print(f"Izlaz: {output_dir}")
    print(f"FPS: {target_fps}, Dimenzije: {target_size}, Max frejmova: {max_frames}")
    print("=" * 60)

    print("\nInicijalizacija MediaPipe HandLandmarker...")
    landmarker = create_hand_landmarker(
        min_detection_confidence=0.2,
        min_presence_confidence=0.2
    )

    stats = {
        "classes": {},
        "total_videos": 0,
        "total_frames_extracted": 0,
        "total_hands_detected": 0
    }

    os.makedirs(output_dir, exist_ok=True)

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        video_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            continue

        print(f"\n--- Klasa: {class_name} ({len(video_files)} videa) ---")

        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        class_stats = {
            "videos": len(video_files),
            "total_frames": 0,
            "detected_frames": 0,
            "video_details": []
        }

        for video_file in sorted(video_files):
            video_path = os.path.join(class_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_name = video_name.replace(".mp4", "").replace(" ", "_")

            video_output_dir = os.path.join(class_output_dir, f"video_{video_name}")
            os.makedirs(video_output_dir, exist_ok=True)

            hand_frames, landmarks, metadata = process_video_frames(
                video_path, landmarker,
                target_fps=target_fps,
                target_size=target_size,
                max_frames=max_frames
            )

            for i, frame in enumerate(hand_frames):
                frame_path = os.path.join(video_output_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame)

            valid_landmarks = [lm for lm in landmarks if lm is not None]
            if valid_landmarks:
                landmarks_path = os.path.join(video_output_dir, "landmarks.npy")
                np.save(landmarks_path, np.array(valid_landmarks))

            class_stats["total_frames"] += metadata["total_frames"]
            class_stats["detected_frames"] += metadata["detected"]
            class_stats["video_details"].append(metadata)

            det_rate = metadata["detection_rate"] * 100
            print(f"  {video_file}: {metadata['detected']}/{metadata['total_frames']} frejmova "
                  f"({det_rate:.0f}%)")

        stats["classes"][class_name] = class_stats
        stats["total_videos"] += class_stats["videos"]
        stats["total_frames_extracted"] += class_stats["total_frames"]
        stats["total_hands_detected"] += class_stats["detected_frames"]

    landmarker.close()

    print("\n" + "=" * 60)
    print("REZULTATI PREPROCESSINGA")
    print("=" * 60)
    print(f"Ukupno videa: {stats['total_videos']}")
    print(f"Ukupno frejmova: {stats['total_frames_extracted']}")
    print(f"Detektovanih šaka: {stats['total_hands_detected']}")
    overall_rate = stats['total_hands_detected'] / max(1, stats['total_frames_extracted']) * 100
    print(f"Procenat detekcije: {overall_rate:.1f}%")
    print()

    for cls, cs in stats["classes"].items():
        rate = cs["detected_frames"] / max(1, cs["total_frames"]) * 100
        print(f"  {cls:12s}: {cs['detected_frames']:4d} frejmova iz {cs['videos']} videa ({rate:.0f}%)")

    stats_path = os.path.join(output_dir, "preprocessing_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistike sačuvane: {stats_path}")

    return stats


def split_dataset(frames_dir, output_path="data_split.json",
                  train_ratio=0.7, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    test_ratio = 1.0 - train_ratio - val_ratio

    print(f"\nPodela dataseta: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")

    split_data = {
        "train": [],
        "val": [],
        "test": [],
        "class_names": CLASS_NAMES,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        },
        "seed": seed
    }

    class_distribution = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(frames_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  UPOZORENJE: Klasa '{class_name}' ne postoji u {frames_dir}")
            continue

        video_dirs = sorted([d for d in os.listdir(class_dir)
                             if os.path.isdir(os.path.join(class_dir, d))])

        if not video_dirs:
            continue

        np.random.shuffle(video_dirs)

        n_videos = len(video_dirs)
        n_train = max(1, int(n_videos * train_ratio))
        n_val = max(1, int(n_videos * val_ratio))
        n_test = n_videos - n_train - n_val

        if n_test <= 0:
            n_test = 1 if n_videos > 2 else 0
            n_val = 1 if n_videos > 1 else 0
            n_train = n_videos - n_val - n_test

        train_videos = video_dirs[:n_train]
        val_videos = video_dirs[n_train:n_train + n_val]
        test_videos = video_dirs[n_train + n_val:]

        for split_name, videos in [("train", train_videos),
                                    ("val", val_videos),
                                    ("test", test_videos)]:
            for video_dir_name in videos:
                video_dir = os.path.join(class_dir, video_dir_name)
                frame_files = sorted([f for f in os.listdir(video_dir)
                                      if f.endswith(('.jpg', '.png'))])

                for frame_file in frame_files:
                    frame_path = os.path.join(class_name, video_dir_name, frame_file)
                    split_data[split_name].append({
                        "path": frame_path,
                        "class": class_name,
                        "class_idx": class_idx,
                        "video": video_dir_name
                    })

                class_distribution[split_name][class_name] += len(frame_files)

    print(f"\n{'Klasa':12s} | {'Train':>7s} | {'Val':>7s} | {'Test':>7s} | {'Ukupno':>7s}")
    print("-" * 55)
    for class_name in CLASS_NAMES:
        tr = class_distribution["train"].get(class_name, 0)
        va = class_distribution["val"].get(class_name, 0)
        te = class_distribution["test"].get(class_name, 0)
        total = tr + va + te
        print(f"{class_name:12s} | {tr:7d} | {va:7d} | {te:7d} | {total:7d}")

    tr_total = sum(class_distribution["train"].values())
    va_total = sum(class_distribution["val"].values())
    te_total = sum(class_distribution["test"].values())
    total = tr_total + va_total + te_total
    print("-" * 55)
    print(f"{'UKUPNO':12s} | {tr_total:7d} | {va_total:7d} | {te_total:7d} | {total:7d}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_path = os.path.join(base_dir, output_path)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)
    print(f"\nSplit metadata sačuvana: {split_path}")

    return split_data


def load_split_data(frames_dir, split_json_path, split_name="train"):
    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    items = split_data[split_name]

    X = []
    y = []

    for item in items:
        img_path = os.path.join(frames_dir, item["path"])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scaled = scale_to_range(rgb)

        X.append(scaled)
        y.append(item["class_idx"])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def convert_output(num_classes):
    return np.eye(num_classes, dtype=np.float32)


def load_video_sequences(frames_dir, split_json_path, split_name="train",
                         seq_length=15):
    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    items = split_data[split_name]

    video_frames = defaultdict(list)
    video_labels = {}

    for item in items:
        video_key = f"{item['class']}_{item['video']}"
        video_frames[video_key].append(item)
        video_labels[video_key] = item["class_idx"]

    X = []
    y = []

    for video_key, frames in video_frames.items():
        frames.sort(key=lambda x: x["path"])

        loaded = []
        for item in frames:
            img_path = os.path.join(frames_dir, item["path"])
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            scaled = scale_to_range(rgb)
            loaded.append(scaled)

        if not loaded:
            continue

        if len(loaded) >= seq_length:
            indices = np.linspace(0, len(loaded) - 1, seq_length, dtype=int)
            sequence = [loaded[i] for i in indices]
        else:
            sequence = loaded + [loaded[-1]] * (seq_length - len(loaded))

        X.append(np.array(sequence))
        y.append(video_labels[video_key])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def run_preprocessing_pipeline(data_dir=None, frames_dir=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if data_dir is None:
        data_dir = os.path.join(base_dir, "data")
    if frames_dir is None:
        frames_dir = os.path.join(base_dir, "frames")

    start_time = time.time()

    stats = preprocess_dataset(
        data_dir=data_dir,
        output_dir=frames_dir,
        target_fps=EXTRACT_FPS,
        target_size=TARGET_SIZE,
        max_frames=30
    )

    split_data = split_dataset(
        frames_dir=frames_dir,
        output_path="data_split.json",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )

    elapsed = time.time() - start_time
    print(f"\nUkupno vreme: {elapsed:.1f}s")
    print("Preprocessing pipeline završen!")

    return stats, split_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASL Recognition - Preprocessing Pipeline")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--frames-dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=EXTRACT_FPS)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--split-only", action="store_true")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(base_dir, "data")
    frames_dir = args.frames_dir or os.path.join(base_dir, "frames")

    if args.split_only:
        split_dataset(frames_dir)
    else:
        EXTRACT_FPS = args.fps
        run_preprocessing_pipeline(data_dir, frames_dir)
