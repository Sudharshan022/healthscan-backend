"""
report_pipeline.py - Medical Report Analysis Pipeline
======================================================
Handles uploaded medical reports (PDF images, lab reports, prescriptions).
Extracts text, identifies conditions, and suggests general wellness guidance.

IMPORTANT MEDICAL DISCLAIMER:
This module provides general informational suggestions only.
It does NOT prescribe medicines or replace a licensed doctor's advice.
Medicine suggestions are general OTC/lifestyle categories only.

To integrate real OCR:
  pip install pytesseract Pillow
  Or use Google Cloud Vision API / AWS Textract for production.
"""

import re
import logging
from typing import Dict, List

log = logging.getLogger(__name__)

# ── Condition knowledge base ──────────────────────────────────────────────────
# Maps detected keywords → condition info + general suggestions
CONDITION_DB = {
    "diabetes": {
        "display": "Diabetes Mellitus",
        "description": "A metabolic condition where blood sugar levels are elevated due to insufficient insulin production or resistance.",
        "severity_keywords": ["HbA1c", "fasting glucose", "postprandial"],
        "diet": [
            "Low glycaemic index foods: oats, barley, legumes",
            "Avoid refined sugar, white rice, maida products",
            "Eat small frequent meals every 3-4 hours",
            "Include bitter gourd (karela), fenugreek seeds in diet",
            "Stay well hydrated with water, avoid sugary drinks",
        ],
        "exercise": [
            "30 min brisk walk daily — most effective for blood sugar control",
            "Resistance training 3x per week",
            "Yoga: Surya Namaskar, Pranayama, Paschimottanasana",
            "Avoid exercising on empty stomach",
        ],
        "general_otc": [
            "Metformin (prescription required — do NOT self-medicate)",
            "Chromium supplements may help insulin sensitivity",
            "Vitamin D3 supplementation (consult doctor)",
            "Alpha lipoic acid for neuropathy support",
        ],
        "specialist": "Endocrinologist / Diabetologist",
        "tests_recommended": ["HbA1c", "Fasting Blood Sugar", "Postprandial Blood Sugar", "Kidney function test"],
        "warning_signs": ["Frequent urination", "Excessive thirst", "Blurred vision", "Slow wound healing"],
    },
    "hypertension": {
        "display": "Hypertension (High Blood Pressure)",
        "description": "A condition where blood pressure in the arteries is persistently elevated, increasing risk of heart disease and stroke.",
        "severity_keywords": ["systolic", "diastolic", "mmHg", "BP"],
        "diet": [
            "DASH diet: fruits, vegetables, whole grains, low-fat dairy",
            "Reduce sodium to less than 2g/day — avoid pickles, papad, namkeen",
            "Increase potassium: banana, sweet potato, spinach",
            "Avoid alcohol and smoking completely",
            "Omega-3 rich foods: flaxseed, walnuts, fish",
        ],
        "exercise": [
            "Aerobic exercise 30 min daily: walking, cycling, swimming",
            "Yoga and meditation to reduce cortisol",
            "Avoid heavy weight lifting — increases BP temporarily",
            "Monitor BP before and after exercise",
        ],
        "general_otc": [
            "Blood pressure medication requires doctor prescription",
            "Coenzyme Q10 supplements may support heart health",
            "Magnesium glycinate (consult doctor for dose)",
            "Garlic extract supplements have mild BP-lowering effects",
        ],
        "specialist": "Cardiologist / General Physician",
        "tests_recommended": ["Complete Blood Count", "Kidney function", "Lipid profile", "ECG", "Echocardiogram"],
        "warning_signs": ["Severe headache", "Chest pain", "Shortness of breath", "Vision problems"],
    },
    "thyroid": {
        "display": "Thyroid Disorder",
        "description": "Dysfunction of the thyroid gland affecting metabolism, energy levels, and overall hormonal balance.",
        "severity_keywords": ["TSH", "T3", "T4", "hypothyroid", "hyperthyroid"],
        "diet": [
            "Iodine-rich foods for hypothyroidism: iodised salt, seafood",
            "Selenium: Brazil nuts, sunflower seeds, eggs",
            "Avoid raw cruciferous vegetables if hypothyroid (broccoli, cabbage)",
            "Gluten-free diet may help autoimmune thyroid conditions",
            "Limit soy products — can interfere with thyroid medication",
        ],
        "exercise": [
            "Regular moderate cardio: 30 min walk or swimming daily",
            "Yoga: Sarvangasana (shoulderstand), Matsyasana (fish pose) for thyroid",
            "Avoid over-exercising if hyperthyroid (elevated heart rate risk)",
            "Strength training supports metabolism in hypothyroidism",
        ],
        "general_otc": [
            "Levothyroxine — prescription only, do NOT self-medicate",
            "Selenium supplements (200mcg/day) — consult doctor",
            "Zinc and Vitamin D support thyroid function",
            "Ashwagandha — adaptogen that supports thyroid (consult doctor first)",
        ],
        "specialist": "Endocrinologist",
        "tests_recommended": ["TSH", "Free T3", "Free T4", "Anti-TPO antibodies", "Thyroid ultrasound"],
        "warning_signs": ["Unexplained weight change", "Extreme fatigue", "Hair loss", "Palpitations"],
    },
    "anaemia": {
        "display": "Anaemia",
        "description": "A condition where the blood lacks enough healthy red blood cells or haemoglobin to carry adequate oxygen to the body's tissues.",
        "severity_keywords": ["haemoglobin", "Hb", "RBC", "ferritin", "iron"],
        "diet": [
            "Iron-rich foods: spinach, beetroot, dates, pomegranate",
            "Vitamin C with iron foods enhances absorption: lemon on dal",
            "Avoid tea/coffee with meals — reduces iron absorption",
            "Include lean meat, fish, poultry for haeme iron",
            "Folate sources: green leafy vegetables, citrus fruits, lentils",
        ],
        "exercise": [
            "Light to moderate exercise only until haemoglobin improves",
            "Avoid high-intensity workouts — oxygen demand increases",
            "Yoga and breathing exercises: Pranayama, Anulom Vilom",
            "Short 20-min walks daily to improve circulation",
        ],
        "general_otc": [
            "Ferrous sulphate / ferrous ascorbate (iron supplements) — doctor prescribed dose",
            "Vitamin B12 injections/tablets if B12 deficiency anaemia",
            "Folic acid 5mg (especially for women of childbearing age)",
            "Avoid self-medicating iron — overdose is dangerous",
        ],
        "specialist": "Haematologist / General Physician",
        "tests_recommended": ["Complete Blood Count", "Serum ferritin", "Serum iron", "Vitamin B12", "Folate levels"],
        "warning_signs": ["Extreme fatigue", "Pale skin", "Shortness of breath", "Rapid heartbeat", "Dizziness"],
    },
    "cholesterol": {
        "display": "High Cholesterol (Dyslipidaemia)",
        "description": "Elevated levels of LDL cholesterol or triglycerides in the blood, increasing risk of heart disease and stroke.",
        "severity_keywords": ["LDL", "HDL", "triglycerides", "lipid", "cholesterol"],
        "diet": [
            "Increase soluble fibre: oats, flaxseed, apples, beans",
            "Avoid trans fats and saturated fats: vanaspati, fried foods",
            "Include healthy fats: olive oil, avocado, nuts",
            "Eat fatty fish twice a week: mackerel, sardines, salmon",
            "Reduce refined carbohydrates and sugar",
        ],
        "exercise": [
            "Aerobic exercise raises HDL (good cholesterol): 40-min brisk walk daily",
            "Cycling, swimming, jogging — all effective",
            "Avoid sedentary lifestyle — get up every hour",
            "Aim for 150 minutes moderate exercise per week",
        ],
        "general_otc": [
            "Statins (atorvastatin, rosuvastatin) — prescription required",
            "Red yeast rice supplements (mild natural statin effect)",
            "Plant sterols/stanols: available in fortified foods",
            "Omega-3 fish oil supplements (2-4g/day for triglycerides)",
        ],
        "specialist": "Cardiologist / General Physician",
        "tests_recommended": ["Full lipid profile", "Liver function test", "Blood glucose", "Cardiac risk assessment"],
        "warning_signs": ["Chest pain", "Xanthomas (yellowish skin deposits)", "Corneal arcus"],
    },
    "kidney": {
        "display": "Kidney / Renal Condition",
        "description": "Kidney disorders affect the body's ability to filter waste and maintain fluid balance.",
        "severity_keywords": ["creatinine", "urea", "GFR", "eGFR", "proteinuria", "kidney"],
        "diet": [
            "Restrict protein intake as per doctor's advice",
            "Low potassium diet if levels elevated: avoid banana, orange, potato",
            "Low phosphorus: avoid dairy, nuts, cola drinks",
            "Limit fluid intake if advised by nephrologist",
            "Reduce sodium — avoid processed and packaged foods",
        ],
        "exercise": [
            "Light to moderate exercise: walking, stretching",
            "Avoid contact sports or heavy lifting",
            "Monitor blood pressure during exercise",
            "Yoga and pranayama for stress reduction",
        ],
        "general_otc": [
            "Do NOT take NSAIDs (ibuprofen, diclofenac) — harmful to kidneys",
            "Avoid herbal supplements without nephrologist approval",
            "Phosphate binders — prescription required",
            "Vitamin D and calcium as prescribed by doctor only",
        ],
        "specialist": "Nephrologist",
        "tests_recommended": ["Serum creatinine", "eGFR", "Urine ACR", "Kidney ultrasound", "Electrolytes"],
        "warning_signs": ["Swollen ankles", "Reduced urine output", "Fatigue", "Foamy urine", "High BP"],
    },
    "liver": {
        "display": "Liver Condition",
        "description": "Liver disorders affect metabolism, detoxification, and the production of essential proteins.",
        "severity_keywords": ["SGPT", "SGOT", "ALT", "AST", "bilirubin", "liver", "hepatitis"],
        "diet": [
            "Completely avoid alcohol — critical for liver recovery",
            "Low-fat diet: avoid fried, oily, processed foods",
            "High-fibre: whole grains, vegetables, fruits",
            "Include turmeric (curcumin) — hepatoprotective properties",
            "Coffee (2 cups/day) has shown liver-protective effects",
        ],
        "exercise": [
            "Moderate exercise reduces liver fat: 30-min walk daily",
            "Avoid exhaustive exercise during active liver inflammation",
            "Yoga: Ardha Matsyendrasana (twisting poses) support liver",
            "Gradual increase in activity as liver function improves",
        ],
        "general_otc": [
            "Silymarin (milk thistle) — hepatoprotective OTC supplement",
            "Avoid paracetamol overdose — toxic to liver",
            "N-Acetyl Cysteine (NAC) — antioxidant for liver support",
            "All supplements should be cleared by hepatologist",
        ],
        "specialist": "Hepatologist / Gastroenterologist",
        "tests_recommended": ["LFT (Liver function test)", "Hepatitis B/C serology", "Fibroscan", "Liver ultrasound"],
        "warning_signs": ["Jaundice (yellow skin/eyes)", "Abdominal swelling", "Dark urine", "Easy bruising"],
    },
    "vitamin": {
        "display": "Vitamin / Mineral Deficiency",
        "description": "Deficiencies in essential vitamins or minerals affecting overall health and bodily functions.",
        "severity_keywords": ["vitamin D", "vitamin B12", "calcium", "deficiency", "folate"],
        "diet": [
            "Vitamin D: 15 min daily sunlight exposure before 10am",
            "Vitamin B12: eggs, dairy, meat, or fortified plant milks",
            "Calcium: dairy, ragi, sesame seeds, green leafy vegetables",
            "Iron: spinach, dates, jaggery, pomegranate",
            "Folate: green leafy vegetables, lentils, citrus fruits",
        ],
        "exercise": [
            "Weight-bearing exercise for bone density (calcium/Vitamin D)",
            "Outdoor exercise maximises Vitamin D synthesis",
            "Regular moderate activity supports nutrient absorption",
        ],
        "general_otc": [
            "Vitamin D3 + K2 combination (1000-2000 IU/day) — safe OTC dose",
            "Vitamin B12: methylcobalamin preferred (1000mcg/day)",
            "Calcium citrate — better absorbed than calcium carbonate",
            "Iron supplements: take with Vitamin C, not with tea/coffee",
        ],
        "specialist": "General Physician / Dietitian",
        "tests_recommended": ["25-OH Vitamin D", "Vitamin B12 serum", "Serum calcium", "Complete Blood Count"],
        "warning_signs": ["Bone pain", "Fatigue", "Muscle weakness", "Tingling in hands/feet", "Frequent infections"],
    },
    "infection": {
        "display": "Infection / Fever",
        "description": "Bacterial, viral, or fungal infection causing systemic symptoms.",
        "severity_keywords": ["WBC", "CRP", "ESR", "infection", "fever", "culture"],
        "diet": [
            "Stay well hydrated: water, ORS, coconut water",
            "Easy-to-digest foods: khichdi, moong dal soup, curd rice",
            "Vitamin C rich foods to boost immunity: amla, citrus",
            "Turmeric milk (haldi doodh) — anti-inflammatory",
            "Avoid heavy, fried, or spicy foods during active infection",
        ],
        "exercise": [
            "Complete rest during active fever — do NOT exercise",
            "Light walking only after fever resolves (below 99°F)",
            "Resume normal activity gradually over 1-2 weeks",
        ],
        "general_otc": [
            "Paracetamol 500mg for fever (follow package dosage)",
            "ORS (Oral Rehydration Solution) for dehydration",
            "Do NOT self-prescribe antibiotics — antibiotic resistance risk",
            "Seek doctor immediately if fever >103°F or persists >3 days",
        ],
        "specialist": "General Physician / Infectious Disease Specialist",
        "tests_recommended": ["CBC with differential", "CRP", "Blood culture", "Urine culture if UTI suspected"],
        "warning_signs": ["Fever above 103°F", "Difficulty breathing", "Severe dehydration", "Altered consciousness"],
    },
}

# Fallback for unknown reports
GENERIC_RESPONSE = {
    "display": "General Health Assessment",
    "description": "Your report has been analysed. Please consult a qualified doctor for accurate interpretation of your results.",
    "diet": [
        "Eat a balanced diet with all food groups",
        "Stay hydrated with 8-10 glasses of water daily",
        "Include fruits and vegetables in every meal",
        "Limit processed foods, excess salt, and refined sugar",
    ],
    "exercise": [
        "30 minutes of moderate exercise most days of the week",
        "Include both cardio and strength training",
        "Yoga and meditation for overall wellbeing",
        "Ensure 7-8 hours of quality sleep",
    ],
    "general_otc": [
        "A daily multivitamin is generally safe",
        "Vitamin D3 is commonly deficient in India — worth checking",
        "Do NOT self-medicate with prescription drugs",
    ],
    "specialist": "General Physician",
    "tests_recommended": ["Annual health checkup recommended"],
    "warning_signs": ["Consult a doctor if you experience unusual symptoms"],
}


def extract_keywords_from_text(text: str) -> List[str]:
    """Simple keyword extraction from report text."""
    text_lower = text.lower()
    found = []
    keyword_map = {
        "diabetes":     ["diabetes", "diabetic", "hba1c", "blood sugar", "glucose", "insulin"],
        "hypertension": ["hypertension", "blood pressure", "bp ", "systolic", "diastolic", "mmhg"],
        "thyroid":      ["thyroid", "tsh", "thyroxine", "t3", "t4", "hypothyroid", "hyperthyroid"],
        "anaemia":      ["anaemia", "anemia", "haemoglobin", "hemoglobin", "hb ", "rbc", "ferritin"],
        "cholesterol":  ["cholesterol", "ldl", "hdl", "triglyceride", "lipid", "dyslipid"],
        "kidney":       ["kidney", "renal", "creatinine", "gfr", "egfr", "nephro", "urea"],
        "liver":        ["liver", "hepat", "sgpt", "sgot", "alt", "ast", "bilirubin"],
        "vitamin":      ["vitamin d", "vitamin b12", "b12", "calcium defic", "folate", "deficiency"],
        "infection":    ["infection", "fever", "wbc", "crp", "esr", "culture", "bacteria", "viral"],
    }
    for condition, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(condition)
                break
    return list(set(found))


def parse_report_text(raw_text: str) -> Dict:
    """
    Parse extracted report text and identify conditions.
    
    In production: replace with Google Cloud Vision OCR or AWS Textract
    to extract text from PDF/image reports accurately.
    """
    if not raw_text or len(raw_text.strip()) < 10:
        return {
            "conditions_found": [],
            "primary_condition": None,
            "analysis": GENERIC_RESPONSE,
            "raw_text_length": 0,
        }

    found_conditions = extract_keywords_from_text(raw_text)

    if not found_conditions:
        return {
            "conditions_found": [],
            "primary_condition": "general",
            "analysis": GENERIC_RESPONSE,
            "raw_text_length": len(raw_text),
        }

    # Build combined analysis for all detected conditions
    primary = found_conditions[0]
    analysis = CONDITION_DB.get(primary, GENERIC_RESPONSE)

    # Merge recommendations if multiple conditions
    if len(found_conditions) > 1:
        combined_diet     = list(analysis["diet"])
        combined_exercise = list(analysis["exercise"])
        for cond in found_conditions[1:]:
            extra = CONDITION_DB.get(cond, {})
            combined_diet     += extra.get("diet", [])[:2]
            combined_exercise += extra.get("exercise", [])[:1]
        analysis = dict(analysis)
        analysis["diet"]     = combined_diet[:6]
        analysis["exercise"] = combined_exercise[:5]

    return {
        "conditions_found": found_conditions,
        "primary_condition": primary,
        "analysis": analysis,
        "raw_text_length": len(raw_text),
    }


def build_report_analysis(parsed: Dict, user_text_input: str = "") -> Dict:
    """Build the full structured response for the report analysis endpoint."""
    analysis = parsed["analysis"]
    conditions = parsed["conditions_found"]

    display_names = [CONDITION_DB[c]["display"] for c in conditions if c in CONDITION_DB]
    if not display_names:
        display_names = ["General Health Assessment"]

    return {
        "detected_conditions": display_names,
        "primary_condition_display": analysis.get("display", "General Health Assessment"),
        "description": analysis.get("description", ""),
        "recommendations": {
            "diet":           analysis.get("diet", []),
            "exercise":       analysis.get("exercise", []),
            "general_otc_info": analysis.get("general_otc", []),
        },
        "specialist_to_consult": analysis.get("specialist", "General Physician"),
        "tests_recommended":     analysis.get("tests_recommended", []),
        "warning_signs":         analysis.get("warning_signs", []),
        "medicine_disclaimer": (
            "IMPORTANT: The medicine information above is GENERAL and EDUCATIONAL only. "
            "Do NOT purchase or consume any prescription medicine without a doctor's consultation. "
            "Self-medication is dangerous. Always consult a registered medical practitioner."
        ),
        "disclaimer": (
            "This AI report analysis is for informational purposes only and does NOT constitute "
            "a medical diagnosis. Report results must be interpreted by a qualified doctor in "
            "the context of your full medical history. Emergency: Call 112."
        ),
    }
