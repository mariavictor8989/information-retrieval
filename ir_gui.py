"""
Information Storage and Retrieval - PySpark IR application (GUI)

Run:  python ir_gui.py

1. Click "Browse..." and choose a folder with .txt documents
2. Click "Import & Build Index"  (PySpark builds the Positional Index)
3. Look at the tables in the tabs
4. Type a query, choose a mode, click "Search"
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from ir_engine import IREngine, stop_spark

MONO = ("Consolas", 10)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PySpark Information Retrieval")
        self.geometry("1100x760")
        self.engine = IREngine()
        self.jobs = queue.Queue()
        self.busy = False

        self.status = tk.StringVar(value="Choose a dataset folder to begin.")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(
            fill="x", side="bottom")
        self._build_top()
        self._build_tabs()
        self._build_query()
        self.minsize(900, 600)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._poll)

    def _build_top(self):
        box = ttk.LabelFrame(self, text="1. Dataset")
        box.pack(fill="x", padx=8, pady=(8, 4))
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "education")
        self.folder_var = tk.StringVar(value=default if os.path.isdir(default) else "")
        ttk.Entry(box, textvariable=self.folder_var).pack(side="left", fill="x", expand=True,
                                                          padx=6, pady=6)
        ttk.Button(box, text="Browse...", command=self._browse).pack(side="left", padx=2)
        self.build_btn = ttk.Button(box, text="Import & Build Index", command=self._build)
        self.build_btn.pack(side="left", padx=6)

    def _build_tabs(self):
        box = ttk.LabelFrame(self, text="2. Index and calculation tables")
        box.pack(fill="both", expand=True, padx=8, pady=4)
        self.nb = ttk.Notebook(box)
        self.nb.pack(fill="both", expand=True, padx=4, pady=4)
        self.tabs = {}
        for name in ["Dataset", "Positional Index", "TF (raw)", "TF (weighted)", "DF / IDF",
                     "TF-IDF", "Vector lengths", "Normalized TF-IDF"]:
            frame = ttk.Frame(self.nb)
            text = ScrolledText(frame, font=MONO, wrap="none", state="disabled")
            xbar = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
            text.configure(xscrollcommand=xbar.set)
            text.pack(fill="both", expand=True)
            xbar.pack(fill="x")
            self.nb.add(frame, text=name)
            self.tabs[name] = text

    def _build_query(self):
        box = ttk.LabelFrame(self, text="3. Query")
        box.pack(fill="both", expand=True, padx=8, pady=4)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=6, pady=6)
        ttk.Label(row, text="Query:").pack(side="left")
        self.query_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.query_var, font=("Segoe UI", 11))
        entry.pack(side="left", fill="x", expand=True, padx=6)
        entry.bind("<Return>", lambda _e: self._search())
        ttk.Label(row, text="Mode:").pack(side="left")
        self.mode_var = tk.StringVar(value="Auto")
        ttk.Combobox(row, textvariable=self.mode_var, width=22, state="readonly",
                     values=["Auto", "Term search", "Phrase query", "Boolean (AND / AND NOT)"]
                     ).pack(side="left", padx=6)
        self.search_btn = ttk.Button(row, text="Search", command=self._search)
        self.search_btn.pack(side="left")
        ttk.Label(box, foreground="#555",
                  text="Examples:  student   |   online learning   |   "
                       "online learning AND student assessment   |   "
                       "online learning AND NOT student assessment").pack(anchor="w", padx=8)
        self.result = ScrolledText(box, font=MONO, height=14, wrap="none", state="disabled")
        self.result.pack(fill="both", expand=True, padx=6, pady=6)

    @staticmethod
    def _set(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _browse(self):
        folder = filedialog.askdirectory(title="Select the folder that contains the .txt documents",
                                         initialdir=self.folder_var.get() or None)
        if folder:
            self.folder_var.set(folder)

    def _run_in_background(self, work, done):
        """Run slow Spark work off the UI thread, then call done(result, error) on the UI thread."""
        if self.busy:
            return
        self.busy = True
        self.build_btn.state(["disabled"])
        self.search_btn.state(["disabled"])

        def target():
            try:
                self.jobs.put((done, work(), None))
            except Exception as e:           # reported to the user in the UI thread
                self.jobs.put((done, None, e))
        threading.Thread(target=target, daemon=True).start()

    def _poll(self):
        try:
            while True:
                done, result, error = self.jobs.get_nowait()
                self.busy = False
                self.build_btn.state(["!disabled"])
                self.search_btn.state(["!disabled"])
                done(result, error)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ---------------- actions ----------------
    def _build(self):
        folder = self.folder_var.get().strip()
        if not os.path.isdir(folder):
            messagebox.showwarning("Dataset", "Please choose an existing folder.")
            return
        self.status.set("Starting Spark and building the Positional Index ... (the first run takes a few seconds)")

        def work():
            engine = IREngine()
            engine.load_folder(folder)
            engine.build_index()
            return engine

        def done(engine, error):
            if error:
                self.status.set("Failed to import the dataset.")
                messagebox.showerror("Dataset", str(error))
                return
            self.engine = engine
            e = engine
            self._set(self.tabs["Dataset"], e.report_dataset())
            self._set(self.tabs["Positional Index"], e.report_index())
            self._set(self.tabs["TF (raw)"], e.report_raw_tf())
            self._set(self.tabs["TF (weighted)"], e.report_weighted_tf())
            self._set(self.tabs["DF / IDF"], e.report_df_idf())
            self._set(self.tabs["TF-IDF"], e.report_tfidf())
            self._set(self.tabs["Vector lengths"], e.report_lengths())
            self._set(self.tabs["Normalized TF-IDF"], e.report_normalized())
            self._set(self.result, "")
            hint = "  -  Spark reader unavailable, see Dataset tab" if e.read_error else ""
            self.status.set(f"Index ready: {e.N} documents, {len(e.terms)} distinct terms "
                            f"(files read with: {e.read_mode}).{hint}")

        self._run_in_background(work, done)

    def _search(self):
        if not self.engine.pos_index:
            messagebox.showinfo("Search", "Import a dataset and build the index first.")
            return
        mode = {"Auto": "auto", "Term search": "term", "Phrase query": "phrase",
                "Boolean (AND / AND NOT)": "boolean"}[self.mode_var.get()]
        query = self.query_var.get()
        try:
            res = self.engine.search(query, mode)
        except ValueError as e:
            messagebox.showwarning("Search", str(e))
            return
        self._set(self.result, self.engine.format_result(res))
        n = len(res["final"])
        self.status.set(f"{res['kind']}: {n} matching document(s)."
                        + (f"  Best: {res['ranking'][0]}" if n else ""))

    def _close(self):
        try:
            stop_spark()
        finally:
            self.destroy()


if __name__ == "__main__":
    App().mainloop()
