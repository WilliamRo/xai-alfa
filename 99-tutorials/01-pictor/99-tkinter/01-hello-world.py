import tkinter as tk


root = tk.Tk()

root.bind('q', lambda *args: root.destroy())
root.title('Hello World')

root.mainloop()
