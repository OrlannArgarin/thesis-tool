from PIL import Image
import pytesseract
import cv2
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import MarianMTModel, MarianTokenizer

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

deed_found = False
deed_type = ""
page_count = 0


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
        return False
    else:
        return True


def check_deed(text):
    global deed_found

    if deed_found:
        return True

    # Tokenizing the text
    words = word_tokenize(text)

    # Identify if document is deed or not
    is_deed = keyword_tagging(words)

    if not is_deed:
        return False
    else:
        # Update deed_found to True
        deed_found = True
        return True


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


def identify_type(lines):
    global deed_type

    if deed_type:
        return deed_type

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

    if not deed_type:
        deed_type = "CANNOT BE IDENTIFIED"

    return deed_type


def load_model():
    # Load the fine tuned pre-trained MarianMT model and tokenizer
    model_name = "fine_tuned_model"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-tl")
    return model, tokenizer


def translate_text(model, tokenizer, texts, max_length=128):
    # Find the indices of empty strings
    empty_indices = [i for i, text in enumerate(texts) if text == ""]

    # Remove empty strings from the input texts
    non_empty_texts = [text for text in texts if text != ""]

    inputs = tokenizer(non_empty_texts, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length", return_attention_mask=True)

    # Forward pass
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length)

    # Decode the translated texts
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Insert empty strings back into the translated list
    for index in empty_indices:
        translated_texts.insert(index, "")

    return translated_texts


def process_single_file(image_path, page_count):
    global is_deed
    global deed_found
    global is_first_page

    print("****************************************************************************")

    # Perform OCR on the image
    ocr_result = perform_ocr(image_path)
    f = open("ocr.txt", "w")
    f.write(ocr_result)
    f.close

    # Count words
    is_exceed = word_count(ocr_result)
    if not is_exceed:
        print("THE DOCUMENT YOU HAVE UPLOADED HAS EXCEEDED THE WORD LIMIT.\nPLEASE TRY AGAIN")
        print("****************************************************************************")
        return

    # Check if deed document
    is_deed = check_deed(ocr_result)
    if not is_deed:
        print("THE DOCUMENT YOU HAVE UPLOADED HAS BEEN DETECTED AS NOT A DEED DOCUMENT.\nPLEASE UPLOAD A DEED DOCUMENT")
        print("****************************************************************************")
        return

    # Perform summarization
    summary = summarize(ocr_result)
    with open("summary.txt", "w") as f:
        f.write(summary)

    with open("summary.txt", "r", encoding="latin-1") as file:
        input_text = [line.strip() for line in file]

    # # Idetify deed type and check if deed type is already found
    # deed_type = identify_type(input_text)
    # if deed_type == "FOUND":
    #     print(f"PAGE {page_count}")
    # else:
    #     print(f"TYPE OF DEED: {deed_type}")

    # Identify deed type
    if is_first_page:
        deed_type = identify_type(input_text)
        print(f"TYPE OF DEED: {deed_type}")
        is_first_page = False
    else:
        print(f"PAGE {page_count}")

    print("****************************************************************************")
    print("SUMMARIZED + TRANSLATED:")
    print()

    # Load the model
    translation_model, translation_tokenizer = load_model()

    # Translate the text
    translated_text = translate_text(translation_model, translation_tokenizer, input_text)

    for translated_text in zip(translated_text):
        print(" ".join(translated_text))


def process_images_in_folder(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    page_count = 0

    # Loop through each image file in the folder
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)

        page_count += 1

        # Process the single image file
        process_single_file(image_path, page_count)


def check_files(filename):
    global deed_found, deed_type, page_count, is_first_page

    # Reset global variables
    deed_found = False
    deed_type = ""
    page_count = 0
    is_first_page = True

    # Check if the last 4 characters match ".jpg" or ".png"
    if filename.lower().endswith((".jpg", ".png")):
        process_single_file(filename, page_count=0)
    # Check if the last 5 characters match ".jpeg" or ".webp"
    elif filename.lower().endswith((".jpeg", ".webp")):
        process_single_file(filename, page_count=0)
    else:
        process_images_in_folder(filename)


# Main Driver
if __name__ == "__main__":
    # Test file directories
    path = r"C:\Users\Orlann\Desktop\THESIS TOOL\data\deed1\deed1-1.jpg"
    image_path = r"D:\Orlann\School Works\4th yr programming\tool\data\deed4.jpeg"

    DEED = r"D:\Orlann\School Works\4th yr programming\tool\data\deed3.jpg"
    NOT_DEED = r"D:\Orlann\School Works\4th yr programming\tool\data\sample-doc.jpg"

    MULTIPLE_FILES = r"D:\Orlann\School Works\4th yr programming\tool\data\multiple_files"
    MULTIPLE_FILES_2 = r"C:\Users\Orlann\Desktop\THESIS TOOL\data\deed1"

    check_files(MULTIPLE_FILES_2)
