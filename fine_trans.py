import torch
from transformers import MarianMTModel, MarianTokenizer

def translate_sentence(sentence, model, tokenizer, max_length=128):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )

    input_ids = inputs["input_ids"].squeeze()
    attention_mask = inputs["attention_mask"].squeeze()

    # Forward pass
    outputs = model.generate(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), max_length=max_length)

    # Decode the generated tokens
    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_sentence

# Load the fine-tuned model and tokenizer
fine_tuned_model_path = "fine_tuned_model"
model = MarianMTModel.from_pretrained(fine_tuned_model_path)
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-tl")  # Use the base model's tokenizer

# Input English sentence for translation
english_sentence = "This Deed of Absolute Sale is made and executed by"

# Translate the sentence
translated_sentence = translate_sentence(english_sentence, model, tokenizer)

# Print the results
print("Input (English):", english_sentence)
print("Translated (Filipino/Tagalog):", translated_sentence)
