import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./emotion_model")
tokenizer = AutoTokenizer.from_pretrained("./emotion_model")

# Function to predict emotion from text
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    emotion_mapping = {0: 'anxiety', 1: 'depression', 2: 'hope', 3: 'happy', 4: 'insomnia'}
    return emotion_mapping[predicted_class]

# Test with a sample text
sample_text = "It’s like living in a world where all the colors have faded, and I’m stuck in a dull, gray fog that never lifts. Conversations feel like a chore, and even the things I used to love feel empty and pointless now. I put on a smile for others, but it’s just a mask that hides the exhaustion I feel deep inside. Every day is just another hurdle, another battle with a heaviness that words can’t fully describe."
print(f"Predicted emotion: {predict_emotion(sample_text)}")
