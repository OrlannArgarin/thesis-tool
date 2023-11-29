# Command prompt run as administrator
# pip install googletrans==4.0.0-rc1

import tkinter as tk
from googletrans import Translator

class LanguageTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Translator")
        
        self.label = tk.Label(root, text="Enter text in English:")
        self.label.pack(pady=10)
        
        self.entry = tk.Text(root, width=50, height=10)
        self.entry.pack(pady=10)
        
        self.translate_button = tk.Button(root, text="Translate", command=self.translate)
        self.translate_button.pack(pady=10)
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

    def translate(self):
        text_to_translate = self.entry.get("1.0", tk.END)  # Retrieve the text from the Text widget
        if text_to_translate.strip():
            translator = Translator()
            translation = translator.translate(text_to_translate, src='en', dest='tl')  # 'en' for English, 'tl' for Filipino
            self.result_label.config(text=f"Translated: {translation.text}")
        else:
            self.result_label.config(text="Please enter text to translate.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageTranslator(root)
    root.mainloop()
