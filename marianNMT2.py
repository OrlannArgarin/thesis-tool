# Command prompt run as administrator
# pip install transformers
# pip install torch torchvision torchaudio
# pip install sentencepiece 
# pip install sacremoses

from transformers import MarianMTModel, MarianTokenizer

def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-tl"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(model, tokenizer, texts):
    # Tokenize the input texts
    input_ids = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    # Perform the translation
    outputs = model.generate(**input_ids)

    # Decode the translated texts
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return translated_texts

if __name__ == "__main__":
    # Load the model
    translation_model, translation_tokenizer = load_model()

    # Read input texts from a file
    file_path = "deed.txt"  # Replace with the actual path to your file

    with open(file_path, "r", encoding="utf-8") as file:
        deed = [line.strip() for line in file]

    # Translate the texts
    translated_texts = translate_text(translation_model, translation_tokenizer, deed)

    # Output the results
    for input_text, translated_text in zip(deed, translated_texts):
        print(f"English: {input_text}")
        print(f"Filipino: {translated_text}")
        print()
