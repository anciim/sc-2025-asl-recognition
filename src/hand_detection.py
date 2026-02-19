import cv2
import numpy as np
import mediapipe as mp
import os


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs, threshold=127):
    _, binary = cv2.threshold(image_gs, threshold, 255, cv2.THRESH_BINARY)
    return binary


def resize_region(region, size=(224, 224)):
    return cv2.resize(region, size, interpolation=cv2.INTER_NEAREST)


def dilate(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def erode(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def get_model_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", "hand_landmarker.task")


def download_hand_model():
    import urllib.request
    model_path = get_model_path()
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        print(f"Preuzimanje hand_landmarker modela...")
        urllib.request.urlretrieve(url, model_path)
        print(f"Model preuzet: {model_path}")
    return model_path


def create_hand_landmarker(min_detection_confidence=0.3, min_presence_confidence=0.3):
    model_path = download_hand_model()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
    )
    return HandLandmarker.create_from_options(options)


def detect_hand(frame, landmarker, target_size=(224, 224), padding=40):
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, None, None

    landmarks = result.hand_landmarks[0]

    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]

    x_min = int(max(0, min(x_coords) - padding))
    y_min = int(max(0, min(y_coords) - padding))
    x_max = int(min(w, max(x_coords) + padding))
    y_max = int(min(h, max(y_coords) + padding))

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    if bbox_w < 10 or bbox_h < 10:
        return None, None, None

    hand_region = frame[y_min:y_max, x_min:x_max]
    hand_resized = resize_region(hand_region, target_size)

    return hand_resized, landmarks, (x_min, y_min, bbox_w, bbox_h)


def detect_hand_fallback(frame, target_size=(224, 224)):
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = dilate(mask, kernel_size=(5, 5), iterations=2)
    mask = erode(mask, kernel_size=(5, 5), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 1000:
        return None, None

    x, y, bw, bh = cv2.boundingRect(largest_contour)

    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    bw = min(w - x, bw + 2 * padding)
    bh = min(h - y, bh + 2 * padding)

    region = frame[y:y+bh, x:x+bw]
    resized = resize_region(region, target_size)

    return resized, (x, y, bw, bh)


def extract_hand_landmarks_as_features(landmarks):
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)


def draw_hand_bbox(frame, bbox, label=None, color=(0, 255, 0)):
    if bbox is None:
        return frame

    x, y, w, h = bbox
    frame_copy = frame.copy()
    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)

    if label:
        cv2.putText(frame_copy, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame_copy


if __name__ == "__main__":
    import sys

    test_video = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "help", "27207.mp4"
    )

    if len(sys.argv) > 1:
        test_video = sys.argv[1]

    print(f"Testiranje detekcije šake na: {test_video}")

    landmarker = create_hand_landmarker()
    cap = cv2.VideoCapture(test_video)

    detected_count = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        hand_crop, landmarks, bbox = detect_hand(frame, landmarker)

        if hand_crop is not None:
            detected_count += 1

    cap.release()
    landmarker.close()

    print(f"Detektovano šaka: {detected_count}/{total_frames} frejmova "
          f"({100*detected_count/max(1,total_frames):.1f}%)")
