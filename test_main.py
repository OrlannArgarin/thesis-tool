from PIL import Image
import pytesseract
import cv2
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
# nltk.download("wordnet")
# nltk.download("punkt")
# nltk.download("maxent_ne_chunker")
# nltk.download("words")
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")

# Set up Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def perform_ocr(image_path):
    print(image_path)
    # Load image
    image = cv2.imread(image_path)

    # Apply Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Perform OCR on the image
    text = pytesseract.image_to_string(threshold)
    return text


def preprocess_text(text):
    # Tokenize, remove stopwords, and lemmatize the text
    tokenizer = RegexpTokenizer(r"\w+")
    lemmatizer = WordNetLemmatizer()

    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stopwords.words("english")]

    return tokens


def named_entity_recognition(text):
    # Perform Named Entity Recognition using NLTK
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    tree = ne_chunk(pos_tags)

    entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity = " ".join([word for word, tag in subtree.leaves()])
            entities.append((entity, subtree.label()))

    return entities


def keyword_tagging(tokens):
    # Perform keyword tagging based on specific criteria
    keywords = []

    # Example: Identify keywords related to property in the Philippines
    property_keywords = [
        "deed",
        "property",
        "land",
        "title",
        "ownership",
        "Philippines",
    ]

    for token in tokens:
        if token in property_keywords:
            keywords.append(token)

    return keywords


if __name__ == "__main__":
    # Example usage
    image_path = "D:/Orlann/School Works/4th yr programming/thesis tool/data/deed3.jpg"

    # Perform OCR
    ocr_result = perform_ocr(image_path)
    print("OCR Result:")
    print(ocr_result)

    # Preprocess text
    preprocessed_text = preprocess_text(ocr_result)
    print("\nPreprocessed Text:")
    print(preprocessed_text)

    # Named Entity Recognition
    entities = named_entity_recognition(ocr_result)
    print("\nNamed Entities:")
    print(entities)

    # Keyword Tagging
    # keywords = keyword_tagging(preprocessed_text)
    # print("\nKeywords:")
    # print(keywords)
