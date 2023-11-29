from PIL import Image
import pytesseract
import cv2


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from transformers import MarianMTModel, MarianTokenizer

# import os
# from nltk.tag import StanfordNERTagger
# from nltk.tokenize import word_tokenize

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def perform_ocr(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Apply Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 187, 255, cv2.THRESH_BINARY)

    # Perform OCR on the image
    text = pytesseract.image_to_string(threshold)
    return text


def summarize(text):
    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Creating a frequency table to keep the score of each word
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text
    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ""
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    # print(summary)
    return summary


def load_model():
    # Load the pre-trained MarianMT model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-tl"  # English to Filipino translation model
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer


def translate_text(model, tokenizer, texts):
    # Filter out empty strings
    non_empty_texts = [text for text in texts if text.strip()]

    if not non_empty_texts:
        return []

    # Tokenize the input texts
    input_ids = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    # Perform the translation
    outputs = model.generate(**input_ids)

    # Decode the translated texts
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return translated_texts


# Main Driver
if __name__ == "__main__":
    # Example usage
    image_path = "D:/Orlann/School Works/4th yr programming/tool/data/deed3.jpg"

    # Perform OCR
    ocr_result = perform_ocr(image_path)
    f = open("ocr.txt", "w")
    f.write(ocr_result)
    f.close
    # print("OCR Result:")
    # print(ocr_result)

    # Summarize extracted text
    f = open("ocr.txt", "w+")
    input_text = f.read()
    f.close

    summary = summarize(input_text)
    f = open("summary.txt", "w")
    f.write(summary)
    f.close

    print("Summarized text:")
    f = open("summary.txt", "r")
    summary_text = f.read()
    print(summary_text)
    f.close
    print("****************************************************************************")

    # f = open("summary.txt", "r", encoding="latin-1")
    # input_text = f.read()
    # f.close

    with open("summary.txt", "r", encoding="latin-1") as file:
        input_text = [line.strip() for line in file]

    cleaned_list = [line for line in input_text if line.strip() != ""]

    # Load the model
    translation_model, translation_tokenizer = load_model()

    # print("=========INPUT TEXT============")
    # print(cleaned_list)
    # print("===============================")

    # Translate the text
    translated_text = translate_text(translation_model, translation_tokenizer, cleaned_list)

    f = open("translated.txt", "w")
    paragraph = " ".join(translated_text)
    f.write(paragraph)
    f.close

    # print(f"Filipino: \n{translated_text}")

    # for translated_text in zip(translated_text):
    # print(translated_text)

    # paragraph = " ".join(translated_text)
    # print(paragraph)
