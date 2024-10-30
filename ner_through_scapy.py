import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy

# Define the model path and mappings for labels
model_path = "emotion_model"  # Update with your actual model path
label_mapping = {
    "LABEL_0": "anxiety",
    "LABEL_1": "depression",
    "LABEL_2": "hope",
    "LABEL_3": "insomnia",
    "LABEL_4": "happy"
}

# Load the tokenizer and model for emotion classification
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
emotion_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=False)

# Load SpaCy's language model
nlp = spacy.load("en_core_web_sm")

# Sample text data to analyze (replace with your own data loading)
texts = [
    "I had the best day ever! I met up with my old friends, we laughed so much, and the weather was just perfect for our picnic. I wish every day could be like this."
]

# Process each text to get emotion and keywords
results = []
for text in texts:
    # Predict the emotion
    emotion = emotion_pipeline(text)[0]['label']
    predicted_emotion = label_mapping.get(emotion, "Unknown")
    
    # Extract keywords
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in {'NOUN', 'ADJ', 'VERB'}]

    # Append results
    results.append({"text": text, "predicted_emotion": predicted_emotion, "keywords": keywords})

# Save to DataFrame and output CSV
results_df = pd.DataFrame(results)
results_df.to_csv('emotion_ner_resultss.csv', index=False)

print("Results saved to 'emotion_ner_results.csv'")
print(results_df.head())
