from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

class TextRequest(BaseModel):
    text: str

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the model and tokenizer
# Load the model and tokenizer
model_path = "plagia_model"  # Ensure this path contains all necessary files
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
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
    import uvicorn
    uvicorn.run("backend_plagia:app", host="0.0.0.0", port=8000)

