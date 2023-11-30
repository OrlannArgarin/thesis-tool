import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

my_w = tk.Tk()
my_w.geometry("920x640")  # Size of the window 
my_w.title('Automated Deed Analysis and Jargon Translation Tool')
my_font1=('times', 12, 'bold')
my_font2=('times', 12)
l1 = tk.Label(my_w,text='Ang tool na ito ay ginagamit sa pagsuri at pagsalin ng deed dokument mula sa wikang Ingles tungo sa wikang Filipino. \nLayunin ng tool na ito na magbigay linaw sa mga malalalim na salita mula sa Ingles upang madaling maunawaan ng lahat.',width=100,font=my_font1)  
#l1.grid(row=1,column=4,columnspan=4)
l1.place(x=15, y=20)

l2 = tk.Label(my_w,text='Pumili ng isang deed dokument na \nsusuriin at isasalin mula sa wikang \nInges tungo sa wikang Filipino.',width=40,font=my_font2)  
l2.place(x=400, y=200)

b1 = tk.Button(my_w, text='Pumili ng dokumento', 
   width=20,command = lambda:upload_file())
#b1.grid(row=2,column=6,columnspan=4)
b1.place(x=640, y=450)

b2 = tk.Button(my_w, text='I-upload ang larawan', 
   width=20,command = lambda:upload_file())
#b2.grid(row=4,column=6,columnspan=4)
b2.place(x=640, y=500)

def upload_file():
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png')]   # type of files to select 
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
    x=50 # image position
    y=100 # image position
    for f in filename:
        img=Image.open(f) # read the image file
        img=img.resize((200,200)) # new width & height
        img=ImageTk.PhotoImage(img)
        e1 =tk.Label(my_w)
        e1.place(y=y,x=x)
        e1.image = img # keep a reference! by attaching it to a widget attribute
        e1['image']=img # Show Image  
        if(x==3): # start new line after third column
            y=y+1# start wtih next row
            x=1    # start with first column
        else:       # within the same row 
            x=x+1 # increase to next column                 
my_w.mainloop()  # Keep the window open