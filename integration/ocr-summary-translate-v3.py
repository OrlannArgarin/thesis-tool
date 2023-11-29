from PIL import Image
import pytesseract
import cv2
import sys
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import MarianMTModel, MarianTokenizer

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


def word_count(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in words if word.isalnum()]

    # Count the number of words
    word_count = len(words)

    if word_count > 1000:
        prompt_user_words_exceed()

    # # Join the words back into a single string
    # cleaned_text = " ".join(words)
    # cleaned_text,

    return word_count


def summarize(text):
    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Identify if document is deed or not
    is_deed = keyword_tagging(words)

    if not is_deed:
        prompt_user_not_deed()

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


def keyword_tagging(tokens):
    words = [token.lower() for token in tokens if token.isalnum()]

    # Perform keyword tagging based on specific criteria
    keywords = []

    # Example: Identify keywords related to property in the Philippines
    property_keywords = [
        "deed",
    ]

    for word in words:
        if word in property_keywords:
            keywords.append(word)

    if keywords:
        return True
    else:
        return False


def prompt_user_not_deed():
    # placeholder
    print("SABING DEED DOCUMENT ANG I SCAN EH HAYS")
    sys.exit(0)


def prompt_user_words_exceed():
    # placeholder
    print("ANDAMING WORDSSSS AAAAAAAAAAAAAAA")
    sys.exit(0)


def identify_type(lines):
    lines_to_search = lines[0:6]
    no_empty = [line for line in lines_to_search if line != ""]
    possible_lines = []
    deed_type = ""

    for item in no_empty:
        clean = re.sub(r"\s*[^a-zA-Z0-9\s]+\s*", "", item)
        clean_lowered = clean.lower()
        possible_lines.append(clean_lowered)

    for item in possible_lines:
        if item == "deed of absolute sale":
            deed_type = item.upper()
            break
        elif item == "deed of assignment and transfer of rights":
            deed_type = item.upper()
            break
        elif item == "deed of donation":
            deed_type = item.upper()
            break
        elif item == "deed of repurchase of land sold under pacto de retro":
            deed_type = item.upper()
            break
        elif item == "deed of sale of motor vehicle":
            deed_type = item.upper()
            break
        elif item == "deed of sale of private agricultural land":
            deed_type = item.upper()
            break
        elif item == "deed of sale of registered land":
            deed_type = item.upper()
            break
        elif item == "deed of sale of registered land under pacto de retro":
            deed_type = item.upper()
            break
        elif item == "deed of sale of unregistered land":
            deed_type = item.upper()
            break
        elif item == "deed of sale with mortage":
            deed_type = item.upper()
            break
        elif item == "deed of sale with assumption of mortgage":
            deed_type = item.upper()
            break
        else:
            deed_type = "CANNOT BE IDENTIFIED"
            break

    return deed_type


def load_model():
    # Load the pre-trained MarianMT model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-tl"  # English to Filipino translation model
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer


def translate_text(model, tokenizer, texts):
    # Find the indices of empty strings
    empty_indices = [i for i, text in enumerate(texts) if text == ""]

    # print(empty_indices)

    # Remove empty strings from the input texts
    non_empty_texts = [text for text in texts if text != ""]

    # Tokenize the non-empty input texts
    input_ids = tokenizer(non_empty_texts, return_tensors="pt", truncation=True, padding=True)

    # Perform the translation
    outputs = model.generate(**input_ids)

    # Decode the translated texts
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Insert empty strings back into the translated list
    for index in empty_indices:
        translated_texts.insert(index, "")

    return translated_texts


def process_single_file(image_path):
    print("****************************************************************************")

    # Perform OCR on the image
    ocr_result = perform_ocr(image_path)
    f = open("ocr.txt", "w")
    f.write(ocr_result)
    f.close

    # Count words
    count = word_count(ocr_result)
    print(f"====================WORD COUNT: {count}====================")
    print()

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
    print()

    with open("summary.txt", "r", encoding="latin-1") as file:
        input_text = [line.strip() for line in file]

    deed_type = identify_type(input_text)
    print(f"====================TYPE OF DEED: {deed_type}====================")

    # Load the model
    translation_model, translation_tokenizer = load_model()

    # Translate the text
    translated_text = translate_text(translation_model, translation_tokenizer, input_text)

    # f = open("translated.txt", "w")

    for translated_text in zip(translated_text):
        print(" ".join(translated_text))
        # line = " ".join(translated_text)
        # f.write(line)
    # print(translated_text)
    # f.close


def process_images_in_folder(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    # Loop through each image file in the folder
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)

        # Process the single image file
        process_single_file(image_path)


def check_files(filename):
    # Check if the last 4 characters match ".jpg" or ".png"
    if filename.lower().endswith((".jpg", ".png")):
        process_single_file(filename)
    # Check if the last 5 characters match ".jpeg" or ".webp"
    elif filename.lower().endswith((".jpeg", ".webp")):
        process_single_file(filename)
    else:
        process_images_in_folder(filename)


# Main Driver
if __name__ == "__main__":
    # Path to deed
    image_path = r"D:\Orlann\School Works\4th yr programming\tool\data\deed4.jpeg"

    DEED = r"D:\Orlann\School Works\4th yr programming\tool\data\deed3.jpg"
    NOT_DEED = r"D:\Orlann\School Works\4th yr programming\tool\data\sample-doc.jpg"

    MULTIPLE_FILES = r"D:\Orlann\School Works\4th yr programming\tool\data\multiple_files"

    # process_images_in_folder(MULTIPLE_FILES)

    check_files(image_path)
