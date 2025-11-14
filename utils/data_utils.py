import json
import os

DOCS_PATH = os.path.join(os.path.dirname(__file__), "../clinic_data/sample_docs.json")
PATIENTS_PATH = os.path.join(os.path.dirname(__file__), "../clinic_data/patient_info.json")

def load_docs():
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_patients():
    with open(PATIENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def search_patient(name):
    patients = load_patients()
    for p in patients:
        if name.lower() in p["name"].lower():
            return p
    return None
