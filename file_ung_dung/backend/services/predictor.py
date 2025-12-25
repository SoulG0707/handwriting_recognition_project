"""Prediction utilities for handwriting date detection using YOLOv8 + MNIST + OCR."""

from __future__ import annotations

import base64
from functools import lru_cache
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np
import pytesseract
from keras import models
from ultralytics import YOLO

from ..config import CONFIDENCE_THRESHOLD, TESSERACT_CMD, resolve_paths

Box = Tuple[int, int, int, int]


pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


@lru_cache(maxsize=1)
def load_yolo() -> YOLO:
    paths = resolve_paths()
    return YOLO(str(paths["yolo"]))


@lru_cache(maxsize=1)
def load_mnist():
    paths = resolve_paths()
    return models.load_model(paths["mnist"], compile=False)


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể đọc ảnh từ dữ liệu tải lên")
    return img


def split_digits(binary_img: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    digit_images = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w > 5 and h > 10:
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w += 2 * padding
            h += 2 * padding
            digit_img = binary_img[y : y + h, x : x + w]
            digit_images.append((x, digit_img))

    digit_images = sorted(digit_images, key=lambda item: item[0])
    return [digit_img for _, digit_img in digit_images]


def preprocess_and_resize_digits(cropped_img: np.ndarray) -> list[np.ndarray]:
    gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    digits = split_digits(binary)
    resized = [cv.resize(d, (28, 28), interpolation=cv.INTER_AREA) for d in digits]
    return resized


def recognize_digits(digit_images: list[np.ndarray]) -> str:
    model = load_mnist()
    recognized = []
    for digit_img in digit_images:
        normalized = digit_img.astype("float32") / 255.0
        normalized = np.expand_dims(normalized, axis=-1)
        normalized = np.expand_dims(normalized, axis=0)
        prediction = model.predict(normalized, verbose=0)
        predicted_digit = int(np.argmax(prediction, axis=1)[0])
        recognized.append(str(predicted_digit))
    return "".join(recognized)


def recognize_digits_from_box(img: np.ndarray, box: Box) -> str:
    x1, y1, x2, y2 = box
    cropped = img[y1:y2, x1:x2]
    digit_images = preprocess_and_resize_digits(cropped)
    if not digit_images:
        return ""
    return recognize_digits(digit_images)


def recognize_text_from_box(img: np.ndarray, box: Box) -> str:
    x1, y1, x2, y2 = box
    cropped = img[y1:y2, x1:x2]
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config="--psm 8 digits")
    return text.strip()


def detect_label_boxes(img: np.ndarray) -> Dict[str, Box]:
    model = load_yolo()
    results = model.predict(source=img, conf=CONFIDENCE_THRESHOLD, verbose=False)
    boxes: Dict[str, Tuple[float, Optional[Box]]] = {
        "ngay": (0.0, None),
        "thang": (0.0, None),
        "nam": (0.0, None),
    }

    class_map = {0: "ngay", 1: "thang", 2: "nam"}

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            cls_id = int(box.cls[0])
            label = class_map.get(cls_id)
            if not label:
                continue
            if conf > boxes[label][0]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes[label] = (conf, (x1, y1, x2, y2))

    return {k: v for k, (_, v) in boxes.items() if v is not None}


def build_value_boxes(label_boxes: Dict[str, Box], width: int) -> Dict[str, Box]:
    ngay = label_boxes.get("ngay")
    thang = label_boxes.get("thang")
    nam = label_boxes.get("nam")

    value_boxes: Dict[str, Box] = {}
    if ngay and (thang or nam):
        right = thang[0] if thang else nam[0]
        value_boxes["day"] = (ngay[2], ngay[1], right, ngay[3])
    if thang and nam:
        value_boxes["month"] = (thang[2], thang[1], nam[0], thang[3])
    elif thang:
        value_boxes["month"] = (thang[2], thang[1], width, thang[3])
    if nam:
        value_boxes["year"] = (nam[2], nam[1], width, nam[3])
    return value_boxes


def annotate(img: np.ndarray, label_boxes: Dict[str, Box], value_boxes: Dict[str, Box]) -> np.ndarray:
    annotated = img.copy()
    colors = {
        "ngay": (255, 0, 0),
        "thang": (0, 255, 0),
        "nam": (0, 165, 255),
        "day": (128, 0, 128),
        "month": (0, 255, 0),
        "year": (0, 165, 255),
    }

    for label, box in label_boxes.items():
        x1, y1, x2, y2 = box
        cv.rectangle(annotated, (x1, y1), (x2, y2), colors[label], 2)
        cv.putText(annotated, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, colors[label], 2)

    for label, box in value_boxes.items():
        x1, y1, x2, y2 = box
        cv.rectangle(annotated, (x1, y1), (x2, y2), colors[label], 2)
        cv.putText(annotated, label, (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, colors[label], 2)
    return annotated


def encode_image(image: np.ndarray) -> str:
    success, buffer = cv.imencode(".png", image)
    if not success:
        raise ValueError("Không thể mã hóa ảnh kết quả")
    return base64.b64encode(buffer).decode("utf-8")


def predict_from_bytes(image_bytes: bytes) -> dict:
    img = decode_image(image_bytes)
    label_boxes = detect_label_boxes(img)
    if not label_boxes:
        raise ValueError("Không phát hiện được nhãn ngày/tháng/năm")

    value_boxes = build_value_boxes(label_boxes, width=img.shape[1])
    if not value_boxes:
        raise ValueError("Không tìm thấy vùng chứa giá trị ngày/tháng/năm")

    day = recognize_digits_from_box(img, value_boxes.get("day")) if "day" in value_boxes else ""
    month = recognize_digits_from_box(img, value_boxes.get("month")) if "month" in value_boxes else ""
    year = recognize_text_from_box(img, value_boxes.get("year")) if "year" in value_boxes else ""

    annotated = annotate(img, label_boxes, value_boxes)
    preview = encode_image(annotated)

    return {
        "day": day,
        "month": month,
        "year": year,
        "label_boxes": label_boxes,
        "value_boxes": value_boxes,
        "annotated_image_base64": preview,
    }
