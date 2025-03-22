import sys
import cv2
import numpy as np
import tensorflow as tf
import os

LABELS = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '=',
    11: '-',
    12: '(',
    13: ')',
    14: '[',
    15: ']',
    16: '{',
    17: '}',
    18: '+',
    19: 'a',
    20: 'b',
    21: 'c',
    22: 'div',
    23: 'm',
    24: 'n',
    25: 'sqrt',
    26: 'times',
    27: 'x',
    28: 'y',
    29: 'z',
    30: '*'
}

MODEL_PATH = "/workspace/training/model/model_22-03-2025_11-27-58.h5"

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded from", model_path)
        return model
    except Exception as e:
        print("Error loading model:", e)
        sys.exit(1)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return processed, gray

def segment_characters(processed_image):
    contours, _ = cv2.findContours(
        processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    segments = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        segments.append({
            'xmin': x,
            'ymin': y,
            'xmax': x + w,
            'ymax': y + h,
            'w': w,
            'h': h
        })
    segments = sorted(segments, key=lambda s: s['xmin'])
    return segments

def crop_and_resize(gray_image, segment):
    margin = 2
    x1 = max(segment['xmin'] - margin, 0)
    y1 = max(segment['ymin'] - margin, 0)
    x2 = segment['xmax'] + margin
    y2 = segment['ymax'] + margin
    crop = gray_image[y1:y2, x1:x2]
    try:
        resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print("Error resizing segment:", e)
        return None
    normalized = resized.astype("float32") / 255.0
    normalized = np.expand_dims(normalized, axis=-1) 
    normalized = np.expand_dims(normalized, axis=0)    
    return normalized

def classify_segment(model, segment_img):
    prediction = model.predict(segment_img)
    index = np.argmax(prediction)
    symbol = LABELS.get(index, "")
    return symbol, prediction

def merge_segments(segments):
    if not segments:
        return segments
    merged = []
    current = segments[0]
    for seg in segments[1:]:
        gap = seg['xmin'] - current['xmax']
        current_center = (current['ymin'] + current['ymax']) / 2
        seg_center = (seg['ymin'] + seg['ymax']) / 2
        if gap < 5 and abs(current_center - seg_center) < 5:
            current['xmax'] = max(current['xmax'], seg['xmax'])
            current['ymin'] = min(current['ymin'], seg['ymin'])
            current['ymax'] = max(current['ymax'], seg['ymax'])
            current['w'] = current['xmax'] - current['xmin']
            current['h'] = current['ymax'] - current['ymin']
        else:
            merged.append(current)
            current = seg
    merged.append(current)
    return merged

def ocr_inference(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: could not load image", image_path)
        sys.exit(1)
    processed, gray = preprocess_image(image)
    segments = segment_characters(processed)
    segments = merge_segments(segments)
    model = load_model(MODEL_PATH)
    recognized_text = ""
    predictions = []
    for seg in segments:
        seg_img = crop_and_resize(gray, seg)
        if seg_img is None:
            continue
        symbol, pred = classify_segment(model, seg_img)
        seg['label'] = symbol
        seg['prediction'] = pred.tolist() 
        predictions.append(seg)
        recognized_text += symbol
    return recognized_text, predictions, image

def main():
    if len(sys.argv) != 2:
        print("Usage: python ocr_inference_improved.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    recognized_text, predictions, image = ocr_inference(image_path)
    print("Recognized Text:", recognized_text)
    processed, _ = preprocess_image(image)
    segments = segment_characters(processed)
    for seg in segments:
        cv2.rectangle(image, (seg['xmin'], seg['ymin']), (seg['xmax'], seg['ymax']), (0, 255, 0), 1)
    cv2.imshow("Segmented Symbols", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
