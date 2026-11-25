"""
Information Retrieval project - command line version

Run:  python ir_cli.py [folder_with_txt_files]

Queries:  word | word1 word2 | A AND B | A AND NOT B
"""
import os
import sys

import ir_engine

MENU = """
1  Dataset                  6  Vector lengths
2  Positional Index         7  Normalized TF-IDF
3  TF (raw and weighted)    8  All tables
4  DF and IDF               9  Search
5  TF-IDF                   0  Quit
"""


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    folder = sys.argv[1] if len(sys.argv) > 1 else input(
        "Dataset folder (Enter = datasets/education): ").strip().strip('"')
    if not folder:
        folder = os.path.join(here, "datasets", "education")

    try:
        print("Reading the documents and building the Positional Index with Spark ...")
        ir = ir_engine.load_dataset(folder)
    except ValueError as error:
        print(error)
        return
    print(f"Done: {ir['N']} documents, {len(ir['terms'])} terms "
          f"(files read with: {ir['read_with']})")

    tables = ir_engine.tables(ir)        # list of (title, text)
    # menu number -> which tables to show
    show = {"1": [0], "2": [1], "3": [2, 3], "4": [4], "5": [5], "6": [6], "7": [7],
            "8": range(8)}
    while True:
        print(MENU)
        choice = input("Choose: ").strip()
        if choice == "0":
            break
        if choice in show:
            for i in show[choice]:
                title, text = tables[i]
                print("=" * 70 + "\n" + title + "\n" + "=" * 70 + "\n" + text + "\n")
        elif choice == "9":
            while True:
                query = input("query (Enter to go back)> ").strip()
                if not query:
                    break
                try:
                    print(ir_engine.result_text(ir_engine.search(ir, query)))
                except ValueError as error:
                    print(error)
        else:
            print("Please choose a number from the menu.")
    ir_engine.stop_spark()


if __name__ == "__main__":
    main()
