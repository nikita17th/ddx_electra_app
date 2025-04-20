import json
from os import environ

import torch
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from transformers import ElectraForSequenceClassification, AutoTokenizer

app = FastAPI()

MONGODB_URL = environ.get("MONGODB_URL")
client = AsyncIOMotorClient(MONGODB_URL)
db = client["survey_db"]
questions_collection = db["questions"]

model = ElectraForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

with open('labels.json', 'r') as f:
    all_labels = json.load(f)

from typing import Union, Dict, List

from pydantic import BaseModel


class QuestionInDB(BaseModel):
    name: str
    code_question: str
    question_en: str
    question_fr: str
    question_ru: str
    is_antecedent: bool
    default_value: Union[str, int, bool]
    value_meaning: dict
    possible_values: list
    data_type: str
    description_en: Union[str, None] = None
    description_fr: Union[str, None] = None
    description_ru: Union[str, None] = None


class InputData(BaseModel):
    AGE: int
    SEX: str
    EVIDENCES: str
    INITIAL_EVIDENCE: str


class Prediction(BaseModel):
    disease: str
    probability: float


class ProfileResponse(BaseModel):
    answers: Dict[str, Union[str, int, bool]]


class ProfileStatusResponse(BaseModel):
    has_profile: bool


@app.delete("/api/profile")
async def delete_profiles():
    await db.profiles.delete_many({})
    return {"status": "success"}


@app.get("/api/profile/status", response_model=ProfileStatusResponse)
async def check_profile_status():
    profile = await db.profiles.find_one(sort=[("_id", -1)])
    return {"has_profile": bool(profile)}


@app.get("/api/profile/questions", response_model=dict)
async def get_profile_questions():
    try:
        questions = await questions_collection.find({"is_antecedent": True}).to_list(length=None)
        for q in questions:
            q["_id"] = str(q["_id"])
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profile", response_model=ProfileResponse)
async def get_profile():
    try:
        profile = await db.profiles.find_one(sort=[("_id", -1)])
        if not profile:
            questions = await questions_collection.find({"is_antecedent": True}).to_list(None)
            default_answers = {q["name"]: q["default_value"] for q in questions}
            await db.profiles.insert_one({"answers": default_answers})
            profile = {"answers": default_answers}

        questions = await questions_collection.find({"is_antecedent": True}).to_list(None)
        for q in questions:
            if q["data_type"] == "B" and isinstance(profile["answers"].get(q["name"]), str):
                profile["answers"][q["name"]] = profile["answers"][q["name"]].lower() == "true"

        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/profile")
async def save_profile(answers: Dict[str, Union[str, int, bool]]):
    try:
        for key, value in answers.items():
            if isinstance(value, str):
                if value.lower() == 'true':
                    answers[key] = True
                elif value.lower() == 'false':
                    answers[key] = False
        await db.profiles.update_one(
            {},
            {"$set": {"answers": answers}},
            upsert=True
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# app.py
@app.get("/api/questions", response_model=dict)
async def get_questions():
    try:
        questions = await questions_collection.find({"is_antecedent": False}).to_list(length=None)
        for q in questions:
            q["_id"] = str(q["_id"])
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_prediction_data(data: dict) -> dict:
    # Извлекаем основные поля
    age = data.get("AGE", 0)
    sex = data.get("GENDER", "unknown")
    processed = {
        "AGE": age,
        "SEX": sex,
        "INITIAL_EVIDENCE": "",
        "EVIDENCES": []
    }

    # Ищем первый True для INITIAL_EVIDENCE
    initial_found = False
    evidence_items = []

    for key, value in data.items():
        if key in ["AGE", "GENDER"]:
            continue  # Пропускаем служебные поля

        if not initial_found and value is True:
            processed["INITIAL_EVIDENCE"] = key
            initial_found = True

        # Обработка значений
        if isinstance(value, bool):
            if value:
                evidence_items.append(key)
        elif isinstance(value, list):
            for item in value:
                evidence_items.append(f"{key}_@_{str(item).strip()}")
        else:
            evidence_items.append(f"{key}_@_{str(value).strip()}")

    # Собираем итоговую строку evidences
    processed["EVIDENCES"] = " ".join(evidence_items)

    # Если не нашли initial evidence, используем первый элемент
    if not processed["INITIAL_EVIDENCE"] and evidence_items:
        processed["INITIAL_EVIDENCE"] = evidence_items[0].split('_@_')[0]

    return processed


@app.post("/api/predict", response_model=List[Prediction])
def predict(data: Dict) -> List[Dict[str, float]]:
    processed_data = process_prediction_data(data.get("answers", {}))

    print(processed_data['AGE'])
    print(processed_data['SEX'])
    print(processed_data['EVIDENCES'])
    text = f"Age: {processed_data['AGE']}, Sex: {processed_data['SEX']}. Evidences: {processed_data['EVIDENCES']} Initial evidence: E_77."

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=299
    )

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        probabilities = torch.sigmoid(outputs.logits).squeeze().tolist()

    predictions = [{"disease": disease, "probability": round(prob, 4)} for disease, prob in
                   zip(all_labels, probabilities)]

    print(predictions)
    return predictions
