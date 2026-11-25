"""
Information Storage and Retrieval - PySpark IR application
CORE ENGINE (shared by the command-line app and the GUI)

Pipeline
  1. Import a folder of .txt documents (any names, any number)
  2. Build the Positional Index with PySpark (RDD transformations)
  3. Compute TF, DF, IDF, TF-IDF, document vector lengths, normalized TF-IDF
  4. Process queries: term search, phrase queries, Boolean AND / AND NOT
  5. Compute query-document similarity (cosine) and rank the results

Nothing is hard-coded: N, the documents and the vocabulary all come from the
folder that the user selects.
"""
import glob
import math
import os
import re
import sys
from urllib.parse import unquote

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable



def find_hadoop_home():
    """
    On Windows, Spark needs winutils.exe + hadoop.dll (in <HADOOP_HOME>\\bin) to read files itself.
    Look for them in the usual places so the user does not have to set anything.
    Returns the HADOOP_HOME folder, or None.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser("~")
    candidates = [os.environ.get("HADOOP_HOME", ""),
                  os.path.join(here, "hadoop"), r"C:\hadoop", r"C:\winutils",
                  r"C:\winutils\hadoop-3.0.0", r"C:\hadoop-3.0.0"]
    for pattern in [os.path.join(home, "Downloads", "*", "hadoop-3*"),
                    os.path.join(home, "Downloads", "*", "*", "hadoop-3*"),
                    os.path.join(home, "Downloads", "hadoop-3*"),
                    os.path.join(home, "Desktop", "*", "hadoop-3*"),
                    os.path.join(home, "Documents", "*", "hadoop-3*")]:
        candidates += sorted(glob.glob(pattern), reverse=True)   # newest version first
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "bin", "winutils.exe")):
            return c
    return None


if sys.platform == "win32":
    _hh = find_hadoop_home()
    if _hh:
        os.environ["HADOOP_HOME"] = _hh
        os.environ["hadoop.home.dir"] = _hh
        os.environ["PATH"] = os.path.join(_hh, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession  # noqa: E402

TOKEN_PATTERN = r"[^\W_]+"


def natural_key(name):
    """Sort d2 before d10, file2 before file10."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def tokenize_text(text):
    return re.findall(TOKEN_PATTERN, text.lower())



# Spark session
_spark = None


def get_spark():
    global _spark
    if _spark is None:
        _spark = (SparkSession.builder.appName("IR_PositionalIndex")
                  .master("local[*]")
                  .config("spark.ui.showConsoleProgress", "false")
                  .getOrCreate())
        _spark.sparkContext.setLogLevel("ERROR")
    return _spark


def stop_spark():
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None


class IREngine:
    def __init__(self):
        self.folder = None
        self.read_mode = None       # how the files were read (for the log)
        self.read_error = ""        # why Spark could not read the folder itself (if it failed)
        self.docs = []              # [(doc_id, text)]
        self.doc_ids = []
        self.N = 0
        self.pos_index = {}         # term -> {doc: [positions]}
        self.terms = []
        self.raw_tf = {}
        self.w_tf = {}
        self.df = {}
        self.idf = {}
        self.tfidf = {}
        self.doc_len = {}
        self.norm = {}

    # 1. Dataset import(PySpark reads the folder)
    def load_folder(self, folder, force_python_read=False):
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found: {folder}")
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if not files:
            raise ValueError(f"No .txt files found in: {folder}")

        sc = get_spark().sparkContext
        docs = None
        if not force_python_read:
            try:
                uri = "file:///" + folder.replace("\\", "/").lstrip("/")
                rdd = sc.wholeTextFiles(uri + "/*.txt")
                rdd = rdd.map(lambda kv: (kv[0].replace("\\", "/").split("/")[-1], kv[1]))
                docs = rdd.collect()          # triggers the read
                docs = [(unquote(name), text) for name, text in docs]
                self.read_mode = "Spark wholeTextFiles"
            except Exception as e:
                docs = None                   # e.g. winutils / hadoop.dll missing on Windows
                lines = [l.strip() for l in str(e).splitlines() if l.strip()]
                pick = next((l for l in lines[1:] if re.search(r"(Exception|Error)", l)),
                            lines[0] if lines else type(e).__name__)
                self.read_error = pick[:200]
        if docs is None:
            # Fallback: read with Python, then hand the documents to Spark
            docs = []
            for path in files:
                with open(path, encoding="utf-8", errors="replace") as f:
                    docs.append((os.path.basename(path), f.read()))
            self.read_mode = "Python read + Spark parallelize"

        docs = [(name[:-4] if name.lower().endswith(".txt") else name, text)
                for name, text in docs]
        docs.sort(key=lambda x: natural_key(x[0]))
        self.folder = folder
        self.docs = docs
        self.doc_ids = [d for d, _ in docs]
        self.N = len(docs)
        return self.N


    # 2. Positional index with PySpark
    def build_index(self):
        sc = get_spark().sparkContext
        pattern = TOKEN_PATTERN

        def tokenize(doc):
            import re
            doc_id, text = doc
            tokens = re.findall(pattern, text.lower())
            return [((t, doc_id), pos) for pos, t in enumerate(tokens, start=1)]

        docs_rdd = sc.parallelize(self.docs, max(1, min(self.N, 8)))
        index_rdd = (
            docs_rdd.flatMap(tokenize)                                  # ((term, doc), position)
                    .groupByKey()                                       # ((term, doc), [positions])
                    .mapValues(lambda ps: sorted(ps))
                    .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))      # (term, (doc, positions))
                    .groupByKey()
                    .mapValues(list)
                    .sortByKey()
        )
        index = {}
        for term, lst in index_rdd.collect():
            lst.sort(key=lambda x: natural_key(x[0]))
            index[term] = {d: ps for d, ps in lst}
        self.pos_index = index
        self.terms = sorted(index)
        self._compute_weights()

    # ------------------------------------------------------------------
    # 3. TF, DF, IDF, TF-IDF, lengths, normalization (from the index)
    # ------------------------------------------------------------------
    def _compute_weights(self):
        ids, terms, N = self.doc_ids, self.terms, self.N
        self.raw_tf = {t: {d: len(self.pos_index[t].get(d, [])) for d in ids} for t in terms}
        self.w_tf = {t: {d: (1 + math.log10(self.raw_tf[t][d]) if self.raw_tf[t][d] > 0 else 0.0)
                         for d in ids} for t in terms}
        self.df = {t: len(self.pos_index[t]) for t in terms}
        self.idf = {t: math.log10(N / self.df[t]) for t in terms}
        self.tfidf = {t: {d: self.w_tf[t][d] * self.idf[t] for d in ids} for t in terms}
        self.doc_len = {d: math.sqrt(sum(self.tfidf[t][d] ** 2 for t in terms)) for d in ids}
        self.norm = {t: {d: (self.tfidf[t][d] / self.doc_len[d] if self.doc_len[d] else 0.0)
                         for d in ids} for t in terms}

    # ------------------------------------------------------------------
    # Reports (plain text, used by both CMD and GUI)
    # ------------------------------------------------------------------
    def report_dataset(self):
        w = max([len(d) for d in self.doc_ids] + [4]) + 2
        lines = [f"{'File':<{w}}{'Words':>6}  Text (first 70 chars)"]
        for d, text in self.docs:
            toks = tokenize_text(text)
            preview = " ".join(text.split())[:70]
            lines.append(f"{d:<{w}}{len(toks):>6}  {preview}")
        lines.append(f"\nN = {self.N} documents, {len(self.terms)} distinct terms"
                     f"  (read with: {self.read_mode})")
        if self.read_error:
            lines.append("Note: Spark could not read the folder itself, so Python read the files "
                         "and Spark built the index.\n"
                         "      To let Spark read the files on Windows, see README.txt (winutils).\n"
                         f"      Reason: {self.read_error}")
        return "\n".join(lines)

    def report_index(self):
        w = max([len(t) for t in self.terms] + [4]) + 2
        lines = []
        for t in self.terms:
            parts = "; ".join(f"{d}: {', '.join(map(str, self.pos_index[t][d]))}"
                              for d in self.pos_index[t])
            lines.append(f"{t:<{w}}{parts}")
        return "\n".join(lines)

    def _matrix(self, m, fmt="{:.4f}"):
        w = max([len(t) for t in self.terms] + [4]) + 2
        cw = max([len(d) for d in self.doc_ids] + [6]) + 2
        lines = [f"{'Term':<{w}}" + "".join(f"{d:>{cw}}" for d in self.doc_ids)]
        for t in self.terms:
            row = "".join(f"{(fmt.format(m[t][d]) if m[t][d] else '0'):>{cw}}" for d in self.doc_ids)
            lines.append(f"{t:<{w}}{row}")
        return "\n".join(lines)

    def report_raw_tf(self):
        return self._matrix(self.raw_tf)

    def report_weighted_tf(self):
        return self._matrix(self.w_tf)

    def report_df_idf(self):
        w = max([len(t) for t in self.terms] + [4]) + 2
        lines = [f"{'Term':<{w}}{'DF':>5}{'IDF':>10}   (IDF = log10(N / DF), N = {self.N})"]
        for t in self.terms:
            lines.append(f"{t:<{w}}{self.df[t]:>5}{self.idf[t]:>10.4f}")
        return "\n".join(lines)

    def report_tfidf(self):
        return self._matrix(self.tfidf)

    def report_lengths(self):
        w = max([len(d) for d in self.doc_ids] + [8]) + 2
        lines = [f"{'Document':<{w}}Vector length"]
        for d in self.doc_ids:
            lines.append(f"{d:<{w}}{self.doc_len[d]:.4f}")
        return "\n".join(lines)

    def report_normalized(self):
        return self._matrix(self.norm)

    def all_reports(self):
        sections = [
            ("ASSIGNED DATASET", self.report_dataset()),
            ("POSITIONAL INDEX", self.report_index()),
            ("RAW TF", self.report_raw_tf()),
            ("WEIGHTED TF: 1 + log10(TF)", self.report_weighted_tf()),
            ("DF and IDF", self.report_df_idf()),
            ("TF-IDF MATRIX", self.report_tfidf()),
            ("DOCUMENT VECTOR LENGTHS", self.report_lengths()),
            ("NORMALIZED TF-IDF MATRIX", self.report_normalized()),
        ]
        bar = "=" * 70
        return "\n\n".join(f"{bar}\n{title}\n{bar}\n{body}" for title, body in sections)

    # ------------------------------------------------------------------
    # 4. Query processing
    # ------------------------------------------------------------------
    def _term_search(self, word):
        return {d: list(ps) for d, ps in self.pos_index.get(word, {}).items()}

    def _phrase_search(self, words):
        if any(w not in self.pos_index for w in words):
            return {}
        candidates = set(self.pos_index[words[0]])
        for w in words[1:]:
            candidates &= set(self.pos_index[w])
        result = {}
        for d in candidates:
            starts = [p for p in self.pos_index[words[0]][d]
                      if all((p + i) in self.pos_index[words[i]][d] for i in range(1, len(words)))]
            if starts:
                result[d] = sorted(starts)
        return dict(sorted(result.items(), key=lambda x: natural_key(x[0])))

    def _operand(self, text):
        words = tokenize_text(text)
        if not words:
            raise ValueError("Empty search term in the query.")
        if len(words) == 1:
            return {"text": " ".join(words), "words": words, "kind": "term",
                    "matches": self._term_search(words[0])}
        return {"text": " ".join(words), "words": words, "kind": "phrase",
                "matches": self._phrase_search(words)}

    def search(self, query, mode="auto"):
        """
        mode: auto | term | phrase | boolean
          term    -> one word
          phrase  -> all words must appear next to each other, in order
          boolean -> operands joined by AND / AND NOT (an operand is a word or a phrase)
          auto    -> boolean if AND / AND NOT is present, else term (1 word) or phrase
        """
        if not self.pos_index:
            raise ValueError("Build the index first (import a dataset folder).")
        q = query.strip()
        if not q:
            raise ValueError("Please type a query.")

        pieces = re.split(r"\s+(AND\s+NOT|AND)\s+", q, flags=re.IGNORECASE)
        has_op = len(pieces) > 1
        if mode == "term":
            if len(tokenize_text(q)) != 1 or has_op:
                raise ValueError("Term search expects exactly one word.")
            pieces = [q]
        elif mode == "phrase":
            if has_op:
                raise ValueError("Phrase mode does not use AND / AND NOT.")
            pieces = [q]
        elif mode == "boolean" and not has_op:
            raise ValueError("Boolean mode needs AND or AND NOT, e.g.  a b AND c d")

        operands = [self._operand(p.strip().strip('"')) for p in pieces[0::2]]
        ops = [re.sub(r"\s+", " ", p.upper()) for p in pieces[1::2]]

        final = set(operands[0]["matches"])
        positive_terms = list(operands[0]["words"])
        for op, operand in zip(ops, operands[1:]):
            if op == "AND":
                final &= set(operand["matches"])
                positive_terms += operand["words"]
            else:                                   # AND NOT
                final -= set(operand["matches"])
        final = sorted(final, key=natural_key)

        qvec = self._query_vector(positive_terms)
        sims = {d: sum(w * self.norm[t][d] for t, w in qvec["normed"].items() if t in self.norm)
                for d in final}
        ranking = sorted(final, key=lambda d: (-sims[d], natural_key(d)))
        kind = ("Boolean query" if ops else
                "Term search" if operands[0]["kind"] == "term" else "Phrase query")
        return {"query": q, "kind": kind, "operands": operands, "ops": ops,
                "final": final, "qvec": qvec, "sims": sims, "ranking": ranking}

    def _query_vector(self, terms):
        counts = {}
        for t in terms:
            counts[t] = counts.get(t, 0) + 1
        wtf = {t: 1 + math.log10(c) for t, c in counts.items()}
        raw = {t: wtf[t] * self.idf.get(t, 0.0) for t in counts}
        length = math.sqrt(sum(v ** 2 for v in raw.values()))
        normed = {t: (v / length if length else 0.0) for t, v in raw.items()}
        return {"wtf": wtf, "raw": raw, "length": length, "normed": normed}

    def format_result(self, res):
        out = []
        bar = "=" * 70
        out.append(f"{bar}\n{res['kind'].upper()}: {res['query']}\n{bar}")
        for i, operand in enumerate(res["operands"]):
            label = "" if i == 0 else f"[{res['ops'][i - 1]}] "
            out.append(f"{label}{operand['kind']} \"{operand['text']}\" -> "
                       + (", ".join(operand["matches"]) or "no match"))
            if operand["kind"] == "phrase" and operand["matches"]:
                n = len(operand["words"])
                for d, starts in operand["matches"].items():
                    rng = ", ".join(f"{s}-{s + n - 1}" for s in starts)
                    out.append(f"      {d}: start positions {', '.join(map(str, starts))}"
                               f"  (token ranges {rng})")
            elif operand["kind"] == "term" and operand["matches"]:
                for d, ps in operand["matches"].items():
                    out.append(f"      {d}: positions {', '.join(map(str, ps))}")
        out.append("")
        out.append("Matching documents: " + (", ".join(res["final"]) if res["final"] else "(none)"))
        qv = res["qvec"]
        out.append(f"\nQuery vector length = {qv['length']:.6f}")
        out.append(f"{'Query term':<14}{'Weighted TF':>12}{'Query TF-IDF':>14}{'Normalized':>12}")
        for t in qv["raw"]:
            out.append(f"{t:<14}{qv['wtf'][t]:>12.4f}{qv['raw'][t]:>14.4f}{qv['normed'][t]:>12.4f}")
        if res["final"]:
            out.append(f"\n{'Matched doc':<14}{'Similarity':>12}")
            for d in res["final"]:
                out.append(f"{d:<14}{res['sims'][d]:>12.4f}")
            out.append("Ranking (highest similarity first): " + " > ".join(res["ranking"]))
        return "\n".join(out)
