import torch
import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.preprocessing import MultiLabelBinarizer


app = FastAPI(title="Test")

CLS_MODEL_PATH = "classifier_full_model"
cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_PATH)
cls_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_PATH)
cls_model.eval()

# если доступен GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cls_model.to(device)

with open(f'{CLS_MODEL_PATH}/final_mlb_encoder.pkl', 'rb') as f:
    mlb: MultiLabelBinarizer = pickle.load(f)

def get_classes(texts, threshold=0.5):
    if isinstance(texts, str):
        texts = [texts]  # если один текст
    text_ids = [(text, i) for i, text in enumerate(texts)]
    
    predicted_labels = []
    inferenced_ids = set()
    while True:
        batch_ids = [i for _, i in text_ids]
        batch_texts = [t for t, _ in text_ids]
        inputs = cls_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = cls_model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

        # бинаризация по порогу
        preds = (probs > threshold).astype(int)
        batch_labels = list(mlb.inverse_transform(preds))

        for id, labels in zip(batch_ids, batch_labels):
            if len(labels) > 0:
                predicted_labels.append((id, labels))
                inferenced_ids.add(id)
        
        text_ids = [(t, i) for t, i in text_ids if i not in inferenced_ids]

        threshold -= 0.05
        if len(predicted_labels) == len(texts) or threshold < 0.1:
            for _, id in text_ids:
                predicted_labels.append(id, ["other"])
            break

    predicted_labels.sort(key=lambda x: x[0])
    predicted_labels = [l for _, l in predicted_labels]
    return predicted_labels


sentiment_model_name = "blanchefort/rubert-base-cased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    truncation=True,
    max_length=512,
    device=device,
)

sentiment_label_map = {
    "NEGATIVE": "негативный",
    "NEUTRAL": "нейтральный",
    "POSITIVE": "позитивный"
}

def get_sentiment(text: str) -> str:
    if not text or not text.strip():
        return None
    result = sentiment_analyzer(text)[0]["label"]
    return sentiment_label_map.get(result, None)



# ---------------------------
# Схема запроса
# ---------------------------
class PredictRequestDataRow(BaseModel):
    id: int
    text: str

class PredictRequest(BaseModel):
    data: list[PredictRequestDataRow]

# ---------------------------
# Схема ответа
# ---------------------------
class PredictResponseDataRow(BaseModel):
    id: int
    topics: list[str]
    sentiments: list[str]

class PredictResponse(BaseModel):
    predictions: list[PredictResponseDataRow]


# ---------------------------
# Эндпоинт POST /api/predict
# ---------------------------
@app.post("/predict", response_model=PredictResponse)
async def app_predict(request: PredictRequest):
    data = request.data
    ids = [row.id for row in request.data]
    texts = [row.text for row in request.data]

    classes = get_classes(texts)
    sentiments = [get_sentiment(text) for text in texts]

    response = PredictResponse(
        predictions=[
            PredictResponseDataRow(id=id, topics=topics, sentiments=[sentiment] * max(len(topics), 1))
            for id, topics, sentiment in zip(ids, classes, sentiments)
        ]
    )

    return response

# ---------------------------
# Запуск: python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
# ---------------------------
