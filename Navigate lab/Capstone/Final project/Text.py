import time
import os
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_text_model(model_name="shanover/symps_disease_bert_v3_c41"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return tokenizer, model, quantized


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def predict_with_metrics(text, tokenizer, model):
    model.config.id2label = {
        0: "(Vertigo) Paroxysmal Positional Vertigo", 1: "AIDS", 2: "Acne", 3: "Alcoholic hepatitis",
        4: "Allergy", 5: "Arthritis", 6: "Bronchial Asthma", 7: "Cervical spondylosis", 8: "Chicken pox",
        9: "Chronic cholestasis", 10: "Common Cold", 11: "Dengue", 12: "Diabetes",
        13: "Dimorphic hemorrhoids (piles)", 14: "Drug Reaction", 15: "Fungal infection", 16: "GERD",
        17: "Gastroenteritis", 18: "Heart attack", 19: "Hepatitis B", 20: "Hepatitis C",
        21: "Hepatitis D", 22: "Hepatitis E", 23: "Hypertension", 24: "Hyperthyroidism",
        25: "Hypoglycemia", 26: "Hypothyroidism", 27: "Impetigo", 28: "Jaundice", 29: "Malaria",
        30: "Migraine", 31: "Osteoarthritis", 32: "Paralysis (brain hemorrhage)",
        33: "Peptic ulcer disease", 34: "Pneumonia", 35: "Psoriasis", 36: "Tuberculosis",
        37: "Typhoid", 38: "Urinary tract infection", 39: "Varicose veins", 40: "Hepatitis A"
    }

    device = torch.device("cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt")
    tokens = len(inputs['input_ids'][0])

    inputs = {k: v.to(device) for k, v in inputs.items()}
    start_mem = get_memory_mb()
    start = time.time()

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    end = time.time()
    end_mem = get_memory_mb()

    latency = end - start
    memory = end_mem - start_mem
    throughput = 1.0 / latency if latency > 0 else 0.0

    return {
        "prediction": model.config.id2label[pred_class],
        "latency": latency,
        "memory": memory,
        "tokens": tokens,
        "throughput": throughput
    }
