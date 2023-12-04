import tkinter as tk
from tkinter import messagebox
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import test_final_v2
import sys

filename = ""

my_w = tk.Tk()
my_w.geometry("920x640")  # Size of the window
my_w.title("Automated Deed Analysis and Jargon Translation Tool")
my_font1 = ("Inter", 12, "bold")
my_font2 = ("Inter", 12)
l1 = tk.Label(
    my_w,
    text="Ang tool na ito ay ginagamit sa pagsuri at pagsalin ng deed dokument mula sa\nwikang Ingles tungo sa wikang Filipino. Layunin ng tool na ito na magbigay linaw sa mga\nmalalalim na salita mula sa Ingles upang madaling maunawaan ng lahat.",
    font=my_font1,
)
# l1.grid(row=1,column=4,columnspan=4)
l1.place(x=100, y=20)

l2 = tk.Label(
    my_w,
    text="Pumili ng isang deed dokument na \nsusuriin at isasalin mula sa wikang\nInges tungo sa wikang Filipino.\nI-check ang box sa baba kung ang\ndeed dokument ay binubuo ng maraming pahina.\nSiguraduhin na ang mga pahinang ito ay\nnasa loob ng isang folder",
    width=40,
    font=my_font2,
)
l2.place(x=480, y=190)

b1 = tk.Button(my_w, text="Pumili ng dokumento/folder", font=my_font2, width=25, command=lambda: select_file())
# b1.grid(row=2,column=6,columnspan=4)
b1.place(x=550, y=450)

b2 = tk.Button(my_w, text="I-upload ang napili", font=my_font2, width=25, command=lambda: upload_file())
# b2.grid(row=4,column=6,columnspan=4)
b2.place(x=550, y=500)

folder_var = tk.BooleanVar()
folder_checkbox = tk.Checkbutton(my_w, text="Maraming pahina ang dokument?", variable=folder_var, font=my_font2)
folder_checkbox.place(x=540, y=400)


def open_file_or_folder():
    f_types = [("Image Files", "*.jpg *.jpeg *.png *.webp")]

    if folder_var.get():
        folder_path = filedialog.askdirectory(title="Pumili ng Folder")
        if folder_path:
            return folder_path
    else:
        file_path = filedialog.askopenfilename(title="Pumili ng Dokumento", filetypes=f_types)
        if file_path:
            return file_path


def image_to_display(filename, x, y):
    img = Image.open(filename)  # read the image file
    img = img.resize((400, 500))  # new width & height
    img = ImageTk.PhotoImage(img)
    e1 = tk.Label(my_w)
    e1.place(y=y, x=x)
    e1.image = img  # keep a reference! by attaching it to a widget attribute
    e1["image"] = img  # Show Image
    if x == 3:  # start new line after third column
        y = y + 1  # start wtih next row
        x = 1  # start with first column
    else:  # within the same row
        x = x + 1  # increase to next column


def select_file():
    global filename
    filename = open_file_or_folder()

    x = 50  # image position
    y = 100  # image position

    if not folder_var.get() and filename:
        image_to_display(filename, x, y)
    elif filename:
        image_to_display(r"C:\Users\Orlann\Desktop\THESIS TOOL\folder_img.png", x, y)


def upload_file():
    global filename

    # Check if a file or folder has been selected
    if not filename:
        messagebox.showerror("Error", "Pumili ng dokumento o folder bago mag-upload.")
        return

    loading_label = tk.Label(my_w, text="Naglo-load...", font=my_font1, foreground="Green")
    loading_label.place(x=630, y=570)

    my_w.update_idletasks()

    # Create a new window for displaying the output
    output_window = tk.Toplevel(my_w)
    output_window.title("Output Window")

    # Create a Text widget for displaying the output
    output_text = tk.Text(output_window, wrap=tk.WORD, width=100, height=40, font=("Inter", 13))
    output_text.pack()

    # Redirect standard output to the Text widget
    sys.stdout = TextRedirector(output_text, "stdout")

    # Call the process_single_file function from test_final_v2 to capture its output
    test_final_v2.check_files(filename)

    loading_label.destroy()

    # Restore standard output
    sys.stdout = sys.__stdout__
    output_text.focus_force()


class TextRedirector:
    def __init__(self, text_widget, tag):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, str):
        self.text_widget.insert(tk.END, str, (self.tag,))
        # self.text_widget.see(tk.END)  # Scroll to the end of the Text widget

    def flush(self):
        pass  # Do nothing for flush


my_w.mainloop()  # Keep the window open
