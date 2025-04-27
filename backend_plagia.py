from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()

# Load model and tokenizer
model_path = "plagia_model"  # Make sure it exists
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_text(req: TextRequest):
    max_length = 512
    if len(tokenizer.tokenize(req.text)) > max_length:
        req.text = " ".join(req.text.split()[:max_length])
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    
    label = "human" if prediction == 0 else "ai"
    confidence = probs[0][prediction].item()

    return {"prediction": label, "confidence": round(confidence, 3)}

if __name__ == "__main__":
    uvicorn.run("backend_plagia:app", host="0.0.0.0", port=8000)
