"""
image_pipeline.py – OpenCV Preprocessing + CNN Inference
=========================================================
Full image processing pipeline:
  1. Decode raw bytes → NumPy array (via OpenCV)
  2. Noise reduction (Gaussian blur)
  3. Standardise resolution (224×224 for MobileNetV2 / ResNet50)
  4. Color space conversions to highlight erythema / lesions
  5. Normalization to [0,1] float tensor
  6. CNN inference (placeholder → swap with real .h5 or .pt weights)
  7. Post-processing: probabilities, severity, diet/exercise advice

How to swap in your trained model
──────────────────────────────────
TensorFlow / Keras (.h5):
    model = tf.keras.models.load_model("ml_models/healthscan_v1.h5")

PyTorch (.pt):
    model = torch.load("ml_models/healthscan_v1.pt", map_location="cpu")
    model.eval()

Then replace the body of `_cnn_inference()` with your real forward pass.
"""

import io
import logging
import numpy as np
from typing import Any

import cv2

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SIZE   = (224, 224)       # Standard input for MobileNetV2 / ResNet50
BLUR_KERNEL   = (5, 5)
BLUR_SIGMA    = 0

# ── Class labels (replace with your fine-tuned label set) ────────────────────
LABELS = [
    "Normal / Healthy Skin",
    "Erythema (Redness)",
    "Eczema / Dermatitis",
    "Psoriasis",
    "Acne / Comedones",
    "Urticaria (Hives)",
    "Fungal Infection (Tinea)",
    "Hyperpigmentation",
    "Visible Swelling / Edema",
    "Wound / Abrasion",
]

CATEGORY_MAP = {
    "Normal / Healthy Skin":       "general",
    "Erythema (Redness)":          "dermatology",
    "Eczema / Dermatitis":         "dermatology",
    "Psoriasis":                   "dermatology",
    "Acne / Comedones":            "dermatology",
    "Urticaria (Hives)":           "dermatology",
    "Fungal Infection (Tinea)":    "dermatology",
    "Hyperpigmentation":           "dermatology",
    "Visible Swelling / Edema":    "orthopedics",
    "Wound / Abrasion":            "general_surgery",
}

SEVERITY_THRESHOLDS = {
    "low":    0.50,
    "medium": 0.70,
    "high":   0.85,
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Decode bytes → NumPy BGR image
# ─────────────────────────────────────────────────────────────────────────────

def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR NumPy array via OpenCV."""
    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV could not decode the image. "
                         "Ensure the bytes represent a valid JPEG/PNG.")
    log.debug(f"Decoded image shape: {img.shape}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Noise reduction
# ─────────────────────────────────────────────────────────────────────────────

def _reduce_noise(img: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.
    Preserves edges better than a simple box blur for skin texture analysis.
    """
    return cv2.GaussianBlur(img, BLUR_KERNEL, BLUR_SIGMA)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Standardise resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resize(img: np.ndarray) -> np.ndarray:
    """Resize to TARGET_SIZE using LANCZOS (best quality for downscaling)."""
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Color space analysis
# ─────────────────────────────────────────────────────────────────────────────

def _colour_analysis(img_bgr: np.ndarray) -> dict:
    """
    Convert to multiple color spaces to extract clinically relevant features.

    • HSV  → saturation channel highlights erythema (red inflammation).
    • LAB  → a* channel isolates red–green opponent color (skin lesions).
    • YCrCb → Cr channel sensitive to skin redness.

    Returns a dict of channel means used downstream for feature augmentation.
    """
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    return {
        "hsv_saturation_mean":  float(np.mean(hsv[:, :, 1])),
        "lab_a_channel_mean":   float(np.mean(lab[:, :, 1])),
        "ycrcb_cr_mean":        float(np.mean(ycrcb[:, :, 1])),
        "brightness_mean":      float(np.mean(img_bgr)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Normalise to float tensor
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR→RGB, scale to [0,1], add batch dimension.
    Shape: (1, 224, 224, 3) – ready for TensorFlow/Keras models.
    For PyTorch: transpose to (1, 3, 224, 224) before inference.
    """
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr   = rgb.astype(np.float32) / 255.0
    # MobileNetV2 uses [-1, 1] normalisation (ImageNet style)
    arr   = (arr - 0.5) * 2.0
    return np.expand_dims(arr, axis=0)   # (1, H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – CNN Inference  ★ SWAP THIS WITH YOUR REAL MODEL ★
# ─────────────────────────────────────────────────────────────────────────────

def _cnn_inference(tensor: np.ndarray, colour_features: dict) -> np.ndarray:
    """
    PLACEHOLDER: Returns simulated probabilistic predictions.

    ─────────────────────────────────────────────
    TO USE YOUR OWN TRAINED MODEL
    ─────────────────────────────────────────────
    OPTION A – TensorFlow/Keras (.h5):
        import tensorflow as tf
        _model = tf.keras.models.load_model("ml_models/healthscan_v1.h5")
        probs  = _model.predict(tensor, verbose=0)[0]
        return probs

    OPTION B – PyTorch (.pt):
        import torch
        _model = torch.load("ml_models/healthscan_v1.pt", map_location="cpu")
        _model.eval()
        t = torch.from_numpy(tensor.transpose(0, 3, 1, 2))
        with torch.no_grad():
            logits = _model(t)
            probs  = torch.softmax(logits, dim=1).numpy()[0]
        return probs

    OPTION C – ONNX Runtime (framework-agnostic):
        import onnxruntime as ort
        sess  = ort.InferenceSession("ml_models/healthscan_v1.onnx")
        input_name = sess.get_inputs()[0].name
        probs = sess.run(None, {input_name: tensor})[0][0]
        return probs
    ─────────────────────────────────────────────
    """
    log.info("⚠️  Using PLACEHOLDER CNN inference. "
             "Swap _cnn_inference() body with your real model.")

    # Generate plausible-looking pseudo-probabilities using colour heuristics
    rng    = np.random.default_rng(seed=int(colour_features["lab_a_channel_mean"]))
    raw    = rng.dirichlet(np.ones(len(LABELS)) * 0.8)

    # Boost erythema / eczema likelihood if high saturation detected
    if colour_features["hsv_saturation_mean"] > 100:
        raw[1] *= 2.5   # Erythema
        raw[2] *= 1.8   # Eczema

    # Re-normalise to sum to 1
    probs = raw / raw.sum()
    return probs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Post-processing & report generation
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDATIONS_DB = {
    "Normal / Healthy Skin": {
        "diet":     ["Stay hydrated (8+ glasses/day)", "Eat antioxidant-rich foods (berries, leafy greens)", "Include Omega-3 sources (flaxseed, walnuts, fish)"],
        "exercise": ["Regular 30-min walks", "Yoga for stress reduction", "Adequate sleep (7–9 hrs)"],
        "advice":   "No significant visual abnormality detected. Maintain a healthy skincare routine.",
    },
    "Erythema (Redness)": {
        "diet":     ["Avoid spicy and processed foods", "Increase anti-inflammatory foods: turmeric milk, ginger tea", "Reduce alcohol and caffeine", "Include vitamin C (amla, citrus)"],
        "exercise": ["Gentle yoga; avoid overheating", "Swimming (cool water reduces redness)", "Avoid vigorous outdoor exercise in heat"],
        "advice":   "Redness may indicate inflammation or sun damage. Use SPF30+ sunscreen and consult a dermatologist.",
    },
    "Eczema / Dermatitis": {
        "diet":     ["Eliminate potential triggers: dairy, gluten, nuts (track with food diary)", "Probiotics: curd, buttermilk", "Vitamin D: sunlight (10 mins/day) or supplements"],
        "exercise": ["Low-impact activities to avoid sweating", "Wear breathable cotton clothing during exercise", "Shower immediately post-workout with gentle cleanser"],
        "advice":   "Moisturise twice daily with fragrance-free cream. Avoid harsh soaps. Consult a dermatologist for prescription treatment.",
    },
    "Psoriasis": {
        "diet":     ["Anti-inflammatory diet: olive oil, fish, vegetables", "Avoid red meat, refined sugar, alcohol", "Vitamin D supplementation (consult doctor)"],
        "exercise": ["Moderate walking and swimming", "Yoga reduces stress (key trigger)", "Avoid skin trauma during exercise"],
        "advice":   "Psoriasis is a chronic autoimmune condition. Requires dermatologist management. Light therapy may be recommended.",
    },
    "Acne / Comedones": {
        "diet":     ["Low glycaemic index diet", "Reduce dairy and high-sugar foods", "Increase zinc (pumpkin seeds, chickpeas)", "Stay well hydrated"],
        "exercise": ["Regular moderate exercise reduces cortisol", "Cleanse face immediately post-workout", "Avoid helmet/mask pressure on acne areas"],
        "advice":   "Cleanse twice daily with salicylic acid or benzoyl peroxide cleanser. Do not pop or squeeze.",
    },
    "Urticaria (Hives)": {
        "diet":     ["Identify and eliminate allergen triggers", "Keep a food diary", "Avoid shellfish, eggs, nuts if sensitive", "Stay well hydrated"],
        "exercise": ["Avoid exercise-induced triggers (exercise urticaria is a recognised condition)", "Cool down environment before exercise"],
        "advice":   "Hives can indicate an allergic reaction. Consult an allergist/dermatologist urgently if spreading.",
    },
    "Fungal Infection (Tinea)": {
        "diet":     ["Reduce sugar and refined carbs (feed fungal growth)", "Eat probiotic-rich foods: curd, idli, dosa", "Garlic and coconut oil have antifungal properties"],
        "exercise": ["Keep skin dry; change into dry clothes immediately post-exercise", "Wear moisture-wicking fabrics", "Avoid shared towels and mats at gyms"],
        "advice":   "Keep affected area clean and dry. Antifungal cream (clotrimazole/terbinafine) available OTC. Consult doctor if spreading.",
    },
    "Hyperpigmentation": {
        "diet":     ["Vitamin C-rich foods: amla, guava, lemon", "Tomatoes (lycopene reduces pigmentation)", "Reduce processed foods and refined sugars"],
        "exercise": ["Mandatory SPF50+ sunscreen before outdoor exercise", "Prefer indoor or early morning workouts"],
        "advice":   "Sun protection is critical. Niacinamide and vitamin C serums may help. Consult dermatologist for prescription options.",
    },
    "Visible Swelling / Edema": {
        "diet":     ["Reduce sodium intake significantly", "Increase potassium (banana, sweet potato)", "Magnesium-rich foods: spinach, almonds", "Stay hydrated"],
        "exercise": ["Elevate affected limb when resting", "Gentle range-of-motion exercises", "Swimming provides gentle compression", "Avoid prolonged standing"],
        "advice":   "Swelling may indicate injury, infection, or systemic condition. Seek medical evaluation promptly, especially if sudden or severe.",
    },
    "Wound / Abrasion": {
        "diet":     ["High-protein diet for tissue repair (dal, eggs, paneer)", "Vitamin C for collagen synthesis", "Zinc-rich foods for wound healing (seeds, nuts)", "Iron for oxygen transport: leafy greens, dates"],
        "exercise": ["Rest the affected area", "Gentle range of motion only", "Avoid activities that reopen the wound"],
        "advice":   "Clean wound with saline, apply antiseptic, cover with sterile dressing. Watch for signs of infection (increasing redness, warmth, pus). Seek medical care for deep wounds.",
    },
}


def _determine_severity(confidence: float, label: str) -> str:
    if label == "Normal / Healthy Skin":
        return "none"
    if confidence >= SEVERITY_THRESHOLDS["high"]:
        return "high"
    if confidence >= SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _build_report(label: str, confidence: float, severity: str, body_part: str) -> str:
    severity_desc = {
        "none":   "No significant visual abnormality detected.",
        "low":    "Minor visual indicators are present. Monitoring is recommended.",
        "medium": "Moderate visual indicators. Professional evaluation is advised.",
        "high":   "Significant visual indicators detected. Please consult a specialist promptly.",
    }
    return (
        f"AI Visual Assessment Report\n"
        f"{'='*40}\n"
        f"Area Analysed   : {body_part.title()}\n"
        f"Assessment      : {label}\n"
        f"Confidence      : {confidence*100:.1f}%\n"
        f"Severity Level  : {severity.upper()}\n"
        f"\nSummary: {severity_desc[severity]}\n"
        f"\nNext Steps: {RECOMMENDATIONS_DB.get(label, {}).get('advice', 'Consult a medical professional.')}\n"
        f"\n{'='*40}\n"
        f"⚠️  This is an AI Visual Assessment only and does NOT replace medical diagnosis."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(raw_bytes: bytes, body_part: str = "skin") -> dict:
    """
    Execute the complete image analysis pipeline.

    Args:
        raw_bytes : Decrypted image bytes (in-memory, never touch disk).
        body_part : User-specified body area label.

    Returns:
        Structured dict with labels, probabilities, severity, report, recommendations.
    """
    # Preprocessing chain
    img_bgr        = _decode_image(raw_bytes)
    img_denoised   = _reduce_noise(img_bgr)
    img_resized    = _resize(img_denoised)
    colour_feats   = _colour_analysis(img_resized)
    tensor         = _normalise(img_resized)

    log.info(f"Preprocessing complete | colour_features={colour_feats}")

    # Inference
    probs      = _cnn_inference(tensor, colour_feats)
    top_idx    = int(np.argmax(probs))
    top_label  = LABELS[top_idx]
    confidence = float(probs[top_idx])
    severity   = _determine_severity(confidence, top_label)
    category   = CATEGORY_MAP.get(top_label, "general")

    # All probabilities dict
    all_probs = {LABELS[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)}

    # Recommendations
    rec = RECOMMENDATIONS_DB.get(top_label, {})
    recommendations = {
        "diet":     rec.get("diet", []),
        "exercise": rec.get("exercise", []),
        "advice":   rec.get("advice", ""),
    }

    report = _build_report(top_label, confidence, severity, body_part)

    return {
        "top_label":       top_label,
        "confidence":      confidence,
        "all_probabilities": all_probs,
        "severity":        severity,
        "category":        category,
        "report":          report,
        "recommendations": recommendations,
        "colour_analysis": colour_feats,
    }
