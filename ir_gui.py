"""
Information Retrieval project - GUI version

Run:  python ir_gui.py
  1. Browse... and choose a folder with .txt files
  2. Import & Build Index  (PySpark reads the files and builds the Positional Index)
  3. Look at the tables in the tabs
  4. Type a query and press Search
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import ir_engine

FONT = ("Consolas", 10)
ir = None            # the data of the imported dataset


def show(box, text):
    box.config(state="normal")
    box.delete("1.0", "end")
    box.insert("1.0", text)
    box.config(state="disabled")


def browse():
    folder = filedialog.askdirectory(title="Choose the folder with the .txt files")
    if folder:
        folder_var.set(folder)


def import_dataset():
    global ir
    status.set("Please wait: Spark is reading the documents and building the index ...")
    window.update()
    try:
        ir = ir_engine.load_dataset(folder_var.get())
    except ValueError as error:
        status.set("Import failed.")
        messagebox.showerror("Dataset", str(error))
        return
    for title, text in ir_engine.tables(ir):
        show(tab_boxes[title], text)
    show(result_box, "")
    status.set(f"Index ready: {ir['N']} documents, {len(ir['terms'])} terms "
               f"(files read with: {ir['read_with']})")


def run_search(event=None):
    if ir is None:
        messagebox.showinfo("Search", "Import a dataset first.")
        return
    try:
        res = ir_engine.search(ir, query_var.get())
    except ValueError as error:
        messagebox.showwarning("Search", str(error))
        return
    show(result_box, ir_engine.result_text(res))
    status.set(f"{len(res['matching'])} matching document(s)")


def close():
    ir_engine.stop_spark()
    window.destroy()


window = tk.Tk()
window.title("PySpark Information Retrieval")
window.geometry("1100x760")
window.minsize(900, 600)

status = tk.StringVar(value="Choose a dataset folder and press Import & Build Index.")
ttk.Label(window, textvariable=status, relief="sunken", anchor="w").pack(fill="x", side="bottom")

# 1. dataset
top = ttk.LabelFrame(window, text="1. Dataset")
top.pack(fill="x", padx=8, pady=6)
default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "education")
folder_var = tk.StringVar(value=default)
ttk.Entry(top, textvariable=folder_var).pack(side="left", fill="x", expand=True, padx=6, pady=6)
ttk.Button(top, text="Browse...", command=browse).pack(side="left")
ttk.Button(top, text="Import & Build Index", command=import_dataset).pack(side="left", padx=6)

# 2. tables
middle = ttk.LabelFrame(window, text="2. Tables")
middle.pack(fill="both", expand=True, padx=8)
notebook = ttk.Notebook(middle)
notebook.pack(fill="both", expand=True, padx=4, pady=4)
tab_boxes = {}
for title in ["Dataset", "Positional Index", "TF (raw)", "TF (weighted) = 1 + log10(TF)",
              "DF and IDF", "TF-IDF", "Vector lengths", "Normalized TF-IDF"]:
    box = ScrolledText(notebook, font=FONT, wrap="none", state="disabled")
    notebook.add(box, text=title.split(" =")[0])
    tab_boxes[title] = box

# 3. query
bottom = ttk.LabelFrame(window, text="3. Query")
bottom.pack(fill="both", expand=True, padx=8, pady=6)
row = ttk.Frame(bottom)
row.pack(fill="x", padx=6, pady=6)
ttk.Label(row, text="Query:").pack(side="left")
query_var = tk.StringVar()
entry = ttk.Entry(row, textvariable=query_var)
entry.pack(side="left", fill="x", expand=True, padx=6)
entry.bind("<Return>", run_search)
ttk.Button(row, text="Search", command=run_search).pack(side="left")
ttk.Label(bottom, foreground="#555", text="Examples:  student   |   online learning   |   "
          "online learning AND student assessment   |   "
          "online learning AND NOT student assessment").pack(anchor="w", padx=8)
result_box = ScrolledText(bottom, font=FONT, height=14, wrap="none", state="disabled")
result_box.pack(fill="both", expand=True, padx=6, pady=6)

window.protocol("WM_DELETE_WINDOW", close)
window.mainloop()
