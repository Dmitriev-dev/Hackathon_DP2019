from tkinter import *
import tkinter.font as font
from Trainer_worker import Worker

work = Worker()
root = Tk()
myfont = font.Font(family='Helvetica', size=12)
myfont2 = font.Font(family='Helvetica', size=9)

root.title("InNino")
root.resizable(width=False, height=False)
btn1 = Button(text="НАЧАТЬ", background="#20bf6b", foreground="#fff", padx="15", pady="4", font=myfont, command=work.run)
btn1.grid(column=0, row=0)
btn2 = Button(text="УПРАЖНЕНИЕ 1", background="#54a0ff", foreground="#fff", padx="15", pady="7", font=myfont2, command=work.run2)
btn2.grid(column=1, row=0)
btn3 = Button(text="УПРАЖНЕНИЕ 2", background="#54a0ff", foreground="#fff", padx="15", pady="7", font=myfont2)
btn3.grid(column=2, row=0)
btn4 = Button(text="УПРАЖНЕНИЕ 3", background="#54a0ff", foreground="#fff", padx="15", pady="7", font=myfont2)
btn4.grid(column=3, row=0)
lbl = Label(root, text="Поработаем?", font=("Helvetica", 30))
lbl.grid(columnspan=4, row=1)

root.mainloop()