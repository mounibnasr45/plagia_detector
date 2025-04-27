from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

class TextRequest(BaseModel):
    text: str

app = FastAPI()

# Load the model and tokenizer
model_path = "checkpoint-36288"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.post("/predict")
async def predict_text(req: TextRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    
    label = "human" if prediction == 0 else "ai"
    confidence = probs[0][prediction].item()

    return {"prediction": label, "confidence": round(confidence, 3)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
