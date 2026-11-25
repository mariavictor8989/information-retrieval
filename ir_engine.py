"""
Information Retrieval project - main functions (PySpark)

1. read_documents()        Spark reads the .txt files of a folder
2. build_index()           Spark builds the Positional Index
3. calculate_weights()     TF, DF, IDF, TF-IDF, vector lengths, normalized TF-IDF
4. search()                term / phrase / Boolean (AND, AND NOT) + similarity + ranking
"""
import glob
import math
import os
import re
import sys
from urllib.parse import unquote

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def set_hadoop_home():
    # On Windows Spark needs winutils.exe and hadoop.dll to read files.
    # Look for a folder like hadoop-3.0.0 (with bin\winutils.exe) in the usual places.
    if sys.platform != "win32":
        return
    home = os.path.expanduser("~")
    places = [os.environ.get("HADOOP_HOME", ""), r"C:\hadoop",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "hadoop")]
    places += glob.glob(os.path.join(home, "Downloads", "*", "hadoop-3*"))
    places += glob.glob(os.path.join(home, "Downloads", "*", "*", "hadoop-3*"))
    for p in places:
        if p and os.path.isfile(os.path.join(p, "bin", "winutils.exe")):
            os.environ["HADOOP_HOME"] = p
            os.environ["PATH"] = os.path.join(p, "bin") + os.pathsep + os.environ["PATH"]
            return


set_hadoop_home()
from pyspark.sql import SparkSession  # noqa: E402

spark = None


def get_sc():
    """Start Spark (once) and return the SparkContext."""
    global spark
    if spark is None:
        spark = SparkSession.builder.appName("IR").master("local[*]").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    return spark.sparkContext


def stop_spark():
    global spark
    if spark is not None:
        spark.stop()
        spark = None


def words_of(text):
    """Lowercase the text and split it into words (punctuation is removed)."""
    return re.findall(r"[^\W_]+", text.lower())


def natural_key(name):
    """So that d2 comes before d10."""
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name.lower())]


# 1. Read the documents
def read_documents(folder):
    """Return ([(doc_id, text), ...], how the files were read)."""
    folder = os.path.abspath(folder)
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        raise ValueError("No .txt files were found in this folder.")
    sc = get_sc()
    try:
        path = "file:///" + folder.replace("\\", "/").lstrip("/") + "/*.txt"
        rdd = sc.wholeTextFiles(path).map(lambda kv: (kv[0].split("/")[-1], kv[1]))
        docs = [(unquote(name), text) for name, text in rdd.collect()]
        how = "Spark wholeTextFiles"
    except Exception:
        # Spark could not read the folder (usually winutils is missing on Windows)
        docs = []
        for f in files:
            with open(f, encoding="utf-8", errors="replace") as fh:
                docs.append((os.path.basename(f), fh.read()))
        how = "Python (Spark could not read the folder)"
    docs = [(os.path.splitext(name)[0], text) for name, text in docs]
    docs.sort(key=lambda d: natural_key(d[0]))
    return docs, how


# 2. Positional index with Spark
def build_index(docs):
    """Return {term: {doc_id: [positions]}} (positions start at 1)."""
    sc = get_sc()

    def tokenize(doc):
        import re
        doc_id, text = doc
        words = re.findall(r"[^\W_]+", text.lower())
        return [((w, doc_id), pos) for pos, w in enumerate(words, start=1)]

    rdd = sc.parallelize(docs)
    rdd = rdd.flatMap(tokenize)                                # ((term, doc), position)
    rdd = rdd.groupByKey().mapValues(sorted)                   # ((term, doc), [positions])
    rdd = rdd.map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))    # (term, (doc, positions))
    rdd = rdd.groupByKey().mapValues(list).sortByKey()         # (term, [(doc, positions), ...])

    index = {}
    for term, items in rdd.collect():
        items.sort(key=lambda x: natural_key(x[0]))
        index[term] = {doc: positions for doc, positions in items}
    return index


# 3. TF, DF, IDF, TF-IDF, lengths, normalization

def calculate_weights(index, doc_ids):
    N = len(doc_ids)
    terms = sorted(index)
    raw_tf, w_tf, tfidf, norm = {}, {}, {}, {}
    df = {t: len(index[t]) for t in terms}
    idf = {t: math.log10(N / df[t]) for t in terms}

    for t in terms:
        raw_tf[t], w_tf[t], tfidf[t] = {}, {}, {}
        for d in doc_ids:
            tf = len(index[t].get(d, []))                  # how many times t is in d
            raw_tf[t][d] = tf
            w_tf[t][d] = 1 + math.log10(tf) if tf > 0 else 0
            tfidf[t][d] = w_tf[t][d] * idf[t]

    # length of each document vector, then divide every weight by it
    lengths = {d: math.sqrt(sum(tfidf[t][d] ** 2 for t in terms)) for d in doc_ids}
    for t in terms:
        norm[t] = {d: tfidf[t][d] / lengths[d] for d in doc_ids}

    return {"N": N, "terms": terms, "raw_tf": raw_tf, "w_tf": w_tf, "df": df, "idf": idf,
            "tfidf": tfidf, "lengths": lengths, "norm": norm}


def load_dataset(folder):
    """Do steps 1-3 and return everything in one dictionary."""
    docs, how = read_documents(folder)
    doc_ids = [d for d, _ in docs]
    index = build_index(docs)
    ir = calculate_weights(index, doc_ids)
    ir.update({"docs": docs, "doc_ids": doc_ids, "index": index, "read_with": how})
    return ir



# Tables as text
def matrix_text(ir, name):
    terms, doc_ids = ir["terms"], ir["doc_ids"]
    w = max(len(t) for t in terms) + 2
    cw = max(len(d) for d in doc_ids) + 6
    lines = ["Term".ljust(w) + "".join(d.rjust(cw) for d in doc_ids)]
    for t in terms:
        row = ""
        for d in doc_ids:
            value = ir[name][t][d]
            row += (f"{value:.4f}" if value else "0").rjust(cw)
        lines.append(t.ljust(w) + row)
    return "\n".join(lines)


def tables(ir):
    """All tables of the project as a list of (title, text)."""
    index, terms = ir["index"], ir["terms"]
    w = max(len(t) for t in terms) + 2

    dataset = ["File".ljust(8) + "Words  Text"]
    for d, text in ir["docs"]:
        dataset.append(d.ljust(8) + str(len(words_of(text))).rjust(5) + "  " + " ".join(text.split())[:70])
    dataset.append(f"\nN = {ir['N']} documents, {len(terms)} distinct terms "
                   f"(files read with: {ir['read_with']})")

    positional = []
    for t in terms:
        parts = [f"{d}: {', '.join(map(str, index[t][d]))}" for d in index[t]]
        positional.append(t.ljust(w) + "; ".join(parts))

    df_idf = ["Term".ljust(w) + "DF".rjust(4) + "IDF".rjust(10) + f"   (IDF = log10({ir['N']} / DF))"]
    for t in terms:
        df_idf.append(t.ljust(w) + str(ir["df"][t]).rjust(4) + f"{ir['idf'][t]:.4f}".rjust(10))

    lengths = ["Document   Vector length"]
    for d in ir["doc_ids"]:
        lengths.append(d.ljust(11) + f"{ir['lengths'][d]:.4f}")

    return [("Dataset", "\n".join(dataset)),
            ("Positional Index", "\n".join(positional)),
            ("TF (raw)", matrix_text(ir, "raw_tf")),
            ("TF (weighted) = 1 + log10(TF)", matrix_text(ir, "w_tf")),
            ("DF and IDF", "\n".join(df_idf)),
            ("TF-IDF", matrix_text(ir, "tfidf")),
            ("Vector lengths", "\n".join(lengths)),
            ("Normalized TF-IDF", matrix_text(ir, "norm"))]


# 4. Search
def find(ir, words):
    """Documents where the words appear next to each other, in this order.
    Returns {doc_id: [start positions]}. One word = normal term search."""
    index = ir["index"]
    if any(w not in index for w in words):
        return {}
    docs = set(index[words[0]])
    for w in words[1:]:
        docs &= set(index[w])                       # documents that contain all the words
    found = {}
    for d in docs:
        starts = []
        for p in index[words[0]][d]:
            # word number i must be at position p + i
            if all((p + i) in index[words[i]][d] for i in range(1, len(words))):
                starts.append(p)
        if starts:
            found[d] = starts
    return dict(sorted(found.items(), key=lambda x: natural_key(x[0])))


def search(ir, query):
    """
    Queries:   word            term search
               word1 word2     phrase query
               A AND B         Boolean AND   (A and B are words or phrases)
               A AND NOT B     Boolean AND NOT
    """
    parts = re.split(r"\s+(AND NOT|AND)\s+", query.strip(), flags=re.IGNORECASE)
    if len(parts) not in (1, 3):
        raise ValueError("Use only one operator:  A AND B   or   A AND NOT B")
    first = words_of(parts[0])
    second = words_of(parts[2]) if len(parts) == 3 else None
    if not first or second == []:
        raise ValueError("Please type a word or a phrase on each side of the operator.")

    found1 = find(ir, first)
    found2 = find(ir, second) if second else None
    operator = parts[1].upper() if second else None

    # matching documents and the words of the query vector
    if operator == "AND":
        matching = set(found1) & set(found2)
        query_words = first + second
    elif operator == "AND NOT":
        matching = set(found1) - set(found2)
        query_words = first              # the excluded phrase is not part of the query vector
    else:
        matching = set(found1)
        query_words = first
    matching = sorted(matching, key=natural_key)

    # query vector: weighted TF * IDF, then divide by its length
    weights, counts = {}, {}
    for w in dict.fromkeys(query_words):             # each word once, in query order
        counts[w] = query_words.count(w)
        weights[w] = (1 + math.log10(counts[w])) * ir["idf"].get(w, 0)
    length = math.sqrt(sum(v ** 2 for v in weights.values()))
    normalized = {w: (v / length if length else 0) for w, v in weights.items()}

    # similarity = dot product of the query vector and the normalized document vector
    sims = {}
    for d in matching:
        sims[d] = sum(normalized[w] * ir["norm"][w][d] for w in normalized if w in ir["norm"])
    ranking = sorted(matching, key=lambda d: (-sims[d], natural_key(d)))

    return {"query": query.strip(), "first": first, "second": second, "operator": operator,
            "found1": found1, "found2": found2, "matching": matching, "counts": counts, "weights": weights,
            "length": length, "normalized": normalized, "sims": sims, "ranking": ranking}


def result_text(res):
    def describe(label, words, found):
        kind = "phrase" if len(words) > 1 else "term"
        lines = [f'{label}{kind} "{" ".join(words)}" -> ' + (", ".join(found) or "no match")]
        for d, starts in found.items():
            if len(words) > 1:
                ranges = ", ".join(f"{s}-{s + len(words) - 1}" for s in starts)
                lines.append(f"      {d}: start positions {', '.join(map(str, starts))}"
                             f"  (token ranges {ranges})")
            else:
                lines.append(f"      {d}: positions {', '.join(map(str, starts))}")
        return lines

    bar = "=" * 70
    kind = ("BOOLEAN " + res["operator"]) if res["operator"] else \
           ("PHRASE QUERY" if len(res["first"]) > 1 else "TERM SEARCH")
    out = [bar, f"{kind}: {res['query']}", bar]
    out += describe("", res["first"], res["found1"])
    if res["second"]:
        out += describe(f"[{res['operator']}] ", res["second"], res["found2"])
    out.append("")
    out.append("Matching documents: " + (", ".join(res["matching"]) or "(none)"))
    out.append(f"\nQuery vector length = {res['length']:.6f}")
    out.append("Query term".ljust(14) + "Weighted TF".rjust(12) + "Query TF-IDF".rjust(14) + "Normalized".rjust(12))
    for w in res["weights"]:
        out.append(w.ljust(14) + f"{1 + math.log10(res['counts'][w]):.4f}".rjust(12)
                   + f"{res['weights'][w]:.4f}".rjust(14) + f"{res['normalized'][w]:.4f}".rjust(12))
    if res["matching"]:
        out.append("\n" + "Matched doc".ljust(14) + "Similarity".rjust(12))
        for d in res["matching"]:
            out.append(d.ljust(14) + f"{res['sims'][d]:.4f}".rjust(12))
        out.append("Ranking (highest similarity first): " + " > ".join(res["ranking"]))
    return "\n".join(out)
