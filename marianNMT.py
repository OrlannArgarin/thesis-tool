# Command prompt run as administrator
# pip install transformers
# pip install torch torchvision torchaudio
# pip install sentencepiece 
# pip install sacremoses


from transformers import MarianMTModel, MarianTokenizer

def load_model():
    # Load the pre-trained MarianMT model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-tl"  # English to Filipino translation model
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(model, tokenizer, text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Perform the translation
    outputs = model.generate(**inputs)

    # Decode the translated text
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translated_text

if __name__ == "__main__":
    # Load the model
    translation_model, translation_tokenizer = load_model()

    # Input English text for translation
    input_text = "That the SELLER is the registered owner of a certain parcel of land"

    # Translate the text
    translated_text = translate_text(translation_model, translation_tokenizer, input_text)

    # Output the result
    print(f"English: {input_text}")
    print(f"Filipino: {translated_text}")
