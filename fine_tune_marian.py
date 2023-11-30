import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Load pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-tl"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Create a sample dataset class
class SampleDataset(Dataset):
    def __init__(self, file_path, max_length=128):
        self.data = self.load_data(file_path)
        self.max_length = max_length

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip().split('\t') for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        if len(line) == 1:
            # If only one value is present, use it as the source text
            src_text = line[0]
            tgt_text = ""  # Set target text as an empty string
        else:
            src_text, tgt_text = line

        inputs = tokenizer(
            src_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",  # Use dynamic padding to handle varying sequence lengths
            return_attention_mask=True  # Add attention_mask
        )
        labels = tokenizer(
            tgt_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",  # Use dynamic padding to handle varying sequence lengths
            return_attention_mask=True  # Add attention_mask
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),  # Add attention_mask
            "labels": labels["input_ids"].squeeze()
        }

# Instantiate the sample dataset
dataset = SampleDataset(file_path="sample_dataset.txt")

# Define hyperparameters
learning_rate = 1e-5
batch_size = 4
num_epochs = 3

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
        inputs = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward pass
        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
