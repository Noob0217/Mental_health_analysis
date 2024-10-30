# from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
# import pandas as pd
# from datasets import Dataset
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the trained model and tokenizer
# model_name = "./emotion_model"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load the test dataset
# test_df = pd.read_csv("test_dataset.csv")
# test_dataset = Dataset.from_pandas(test_df)

# # Tokenize the test dataset
# def tokenize_data(example):
#     return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

# test_dataset = test_dataset.map(tokenize_data, batched=True)
# test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'emotion_label'])

# # Define the evaluation metrics function
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

# # Set up training arguments for evaluation
# training_args = TrainingArguments(
#     output_dir='./results',
#     per_device_eval_batch_size=4,
#     logging_dir='./logs',
# )

# # Initialize Trainer with the loaded model for evaluation
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics,
# )

# # Evaluate the model
# eval_results = trainer.evaluate()

# # Predict on the test dataset to get predictions
# predictions = trainer.predict(test_dataset)
# preds = predictions.predictions.argmax(-1)
# labels = test_dataset['emotion_label']

# # Count the number of predictions for each class
# pred_counts = np.bincount(preds, minlength=5)
# label_counts = np.bincount(labels, minlength=5)

# # Map the indices back to emotion labels
# emotion_mapping = {
#     0: 'anxiety',
#     1: 'depression',
#     2: 'hope',
#     3: 'happy',
#     4: 'insomnia'
# }

# pred_distribution = {emotion_mapping[i]: count for i, count in enumerate(pred_counts)}
# label_distribution = {emotion_mapping[i]: count for i, count in enumerate(label_counts)}

# print("Evaluation results:")
# print(eval_results)
# print("\nPrediction Distribution:", pred_distribution)
# print("Label Distribution:", label_distribution)

# # Plot the distributions
# emotions = list(emotion_mapping.values())
# plt.figure(figsize=(10, 5))
# x = np.arange(len(emotions))
# width = 0.4

# plt.bar(x - width/2, [label_distribution[emotion] for emotion in emotions], width, label='Actual')
# plt.bar(x + width/2, [pred_distribution[emotion] for emotion in emotions], width, label='Predicted')

# plt.xlabel('Emotion')
# plt.ylabel('Count')
# plt.title('Actual vs. Predicted Emotion Distribution')
# plt.xticks(x, emotions)
# plt.legend()
# plt.show()




import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# Step 1: Load the Tokenizer, Model, and Test Data
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained("./emotion_model")
tokenizer = AutoTokenizer.from_pretrained("./emotion_model")

# Step 2: Load Test Dataset
test_df = pd.read_csv("test_dataset.csv")
test_dataset = Dataset.from_pandas(test_df)

# Step 3: Tokenize the Test Data
def tokenize_data(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

test_dataset = test_dataset.map(tokenize_data, batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'emotion_label'])

# Step 4: Initialize the Trainer with the model and tokenizer
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer
)

# Step 5: Make Predictions on the Test Dataset
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(-1)
true_labels = test_dataset['emotion_label']

# Step 6: Calculate Accuracy, Precision, Recall, and F1 Score
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

# Step 7: Print the Metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
