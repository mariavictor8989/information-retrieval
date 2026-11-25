"""
Information Storage and Retrieval - PySpark IR application (command line)

Interactive:      python ir_cli.py [dataset_folder]
Show every table: python ir_cli.py <folder> --all
One query:        python ir_cli.py <folder> --query "online learning AND NOT student assessment"

Query syntax
  word                     term search
  word1 word2              phrase query (words next to each other, in order)
  A AND B                  Boolean AND   (A and B are words or phrases)
  A AND NOT B              Boolean AND NOT
"""
import argparse
import os
import sys

from ir_engine import IREngine, stop_spark

MENU = """
------------------------------------------------------------
 1  Show dataset                 5  Show TF-IDF matrix
 2  Show Positional Index        6  Show document vector lengths
 3  Show TF (raw + weighted)     7  Show normalized TF-IDF matrix
 4  Show DF and IDF              8  Show ALL tables
 9  Search (term / phrase / Boolean AND, AND NOT)
 0  Load another dataset folder   q  Quit
------------------------------------------------------------"""


def load(engine, folder):
    print(f"\nImporting documents from: {folder}")
    n = engine.load_folder(folder)
    print(f"Found {n} .txt documents. Building the Positional Index with PySpark ...")
    engine.build_index()
    print(f"Done: {len(engine.terms)} distinct terms (documents read with: {engine.read_mode}).")


def ask_folder(default):
    while True:
        value = input(f"Dataset folder [{default}]: ").strip().strip('"') or default
        if os.path.isdir(value):
            return value
        print("  That folder does not exist, try again.")


def search_loop(engine):
    print("\nType a query (empty line = back to the menu).")
    print("  examples:  online learning   |   student   |   online learning AND student assessment"
          "   |   online learning AND NOT student assessment")
    while True:
        q = input("query> ").strip()
        if not q:
            return
        try:
            print("\n" + engine.format_result(engine.search(q)) + "\n")
        except ValueError as e:
            print(f"  ! {e}")


def main():
    ap = argparse.ArgumentParser(description="PySpark Information Retrieval application")
    ap.add_argument("folder", nargs="?", help="folder that contains the .txt documents")
    ap.add_argument("--all", action="store_true", help="print every table and exit")
    ap.add_argument("--query", action="append", help="run a query and exit (can be repeated)")
    args = ap.parse_args()

    engine = IREngine()
    folder = args.folder
    if folder is None:
        folder = ask_folder(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "education"))
    try:
        load(engine, folder)

        if args.all or args.query:
            if args.all:
                print("\n" + engine.all_reports())
            for q in args.query or []:
                print("\n" + engine.format_result(engine.search(q)))
            return

        actions = {
            "1": lambda: engine.report_dataset(),
            "2": lambda: engine.report_index(),
            "3": lambda: engine.report_raw_tf() + "\n\n--- Weighted TF: 1 + log10(TF) ---\n"
                         + engine.report_weighted_tf(),
            "4": lambda: engine.report_df_idf(),
            "5": lambda: engine.report_tfidf(),
            "6": lambda: engine.report_lengths(),
            "7": lambda: engine.report_normalized(),
            "8": lambda: engine.all_reports(),
        }
        while True:
            print(MENU)
            choice = input("Choose: ").strip().lower()
            if choice in actions:
                print("\n" + actions[choice]())
            elif choice == "9":
                search_loop(engine)
            elif choice == "0":
                load(engine, ask_folder(engine.folder))
            elif choice in ("q", "quit", "exit"):
                break
            else:
                print("Please choose one of the menu options.")
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n{e}" if str(e) else "\nStopped.")
        sys.exit(1)
    finally:
        stop_spark()


if __name__ == "__main__":
    main()
