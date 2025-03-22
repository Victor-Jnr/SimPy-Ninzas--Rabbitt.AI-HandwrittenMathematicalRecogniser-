import sys
import cv2
import numpy as np
import tensorflow as tf
import pytesseract
from pytesseract import Output

REVERSE_MAPPING = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "-", 11: "(", 12: ")", 13: "[", 14: "]",
    15: "{", 16: "}", 17: "+", 18: "a", 19: "b",
    20: "c", 21: "m", 22: "n", 23: "sqrt", 24: "x",
    25: "y", 26: "z", 27: "neq", 28: ".", 29: "*"
}

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def preprocess_image_for_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   11, 2)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def get_char_boxes(image):
    custom_config = r'--oem 3 --psm 10'
    boxes_str = pytesseract.image_to_boxes(image, config=custom_config)
    h, w = image.shape[:2]
    char_boxes = []
    for line in boxes_str.splitlines():
        parts = line.split(' ')
        if len(parts) == 6:
            char, x1, y1, x2, y2, _ = parts
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            y1_cv = h - y2
            y2_cv = h - y1
            char_boxes.append({'char': char, 'x1': x1, 'y1': y1_cv, 'x2': x2, 'y2': y2_cv})
    return char_boxes

def crop_and_preprocess(gray_image, box):
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    crop = gray_image[y1:y2, x1:x2]
    resized = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    normalized = np.expand_dims(normalized, axis=-1)
    normalized = np.expand_dims(normalized, axis=0)
    return normalized

def decode_prediction(pred):
    pred_index = np.argmax(pred, axis=1)[0]
    return REVERSE_MAPPING.get(pred_index, "")

def main():
    if len(sys.argv) != 2:
        print("Usage: python ocr_inference_improved.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = "/workspace/training/model/model_22-03-2025_11-27-58.h5"
    model = load_model(model_path)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        sys.exit(1)
    
    seg_image = preprocess_image_for_segmentation(image)
    boxes = get_char_boxes(seg_image)
    if not boxes:
        print("No character boxes detected.")
        sys.exit(1)
    
    boxes = sorted(boxes, key=lambda b: b['x1'])
    
    recognized_text = ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for box in boxes:
        char_input = crop_and_preprocess(gray, box)
        pred = model.predict(char_input)
        recognized_char = decode_prediction(pred)
        recognized_text += recognized_char
    
    print("Recognized Text:", recognized_text)

if __name__ == "__main__":
    main()
