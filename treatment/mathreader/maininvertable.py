import sys
import cv2
import numpy as np
import tensorflow as tf

from mathreader.image_processing import preprocessing as preproc

REVERSE_MAPPING = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "-", 11: "(", 12: ")", 13: "[", 14: "]",
    15: "{", 16: "}", 17: "+", 18: "a", 19: "b",
    20: "c", 21: "m", 22: "n", 23: "sqrt", 24: "x",
    25: "y", 26: "z", 27: "neq", 28: ".", 29: "*"
}

def load_model(model_path):
    """Load the trained OCR model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def segment_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_intensity = np.median(gray)
    if median_intensity < 127:
        print("Dark background detected. Inverting image colors...")
        img = cv2.bitwise_not(img)
    
    preprocessor = preproc.ImagePreprocessing(configs={'resize': 'smaller', 'dataset': False})
    symbols, full_processed = preprocessor.treatment(img)
    return symbols, full_processed

def preprocess_symbol(symbol_img):
    if symbol_img.ndim == 2:
        symbol_img = np.expand_dims(symbol_img, axis=-1)
    symbol_img = symbol_img.astype("float32")
    symbol_img = np.expand_dims(symbol_img, axis=0)
    return symbol_img

def decode_prediction(pred):
    """Decodes the model output to its corresponding character using REVERSE_MAPPING."""
    pred_index = np.argmax(pred, axis=1)[0]
    return REVERSE_MAPPING.get(pred_index, "")

def main():
    if len(sys.argv) != 2:
        print("Usage: python ocr_inference_multi.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = "/workspace/training/model/model_22-03-2025_11-27-58.h5"
    
    model = load_model(model_path)
    
    symbols, _ = segment_image(image_path)
    if not symbols:
        print("No symbols detected in the image.")
        sys.exit(1)
    
    symbols = sorted(symbols, key=lambda s: s.get("xmin", 0))
    
    recognized_text = ""
    for sym in symbols:
        symbol_img = sym.get("image")
        if symbol_img is None:
            continue
        symbol_input = preprocess_symbol(symbol_img)
        pred = model.predict(symbol_input)
        symbol = decode_prediction(pred)
        recognized_text += symbol
    
    print("Recognized Text:", recognized_text)
    
    fixed_text = recognized_text.replace("--", "=")
    print("Fixed Text:", fixed_text)

if __name__ == "__main__":
    main()
