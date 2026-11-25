PySpark Information Retrieval project
=====================================
ir_engine.py   the main functions (Spark reads the files and builds the Positional Index,
               then TF / DF / IDF / TF-IDF, normalization, search, similarity, ranking)
ir_cli.py      command-line version          (run_cli.bat)
ir_gui.py      graphical version             (run_gui.bat)
datasets\education             the assigned dataset (d1.txt ... d10.txt)
datasets\sample_other_domain   another dataset (different names and number of files)

Needs: Python 3.10+, Java 17+, pip install pyspark
On Windows Spark also needs winutils.exe + hadoop.dll (folder hadoop-3.0.0\bin).
The program looks for that folder in Downloads, C:\hadoop and next to ir_engine.py.
If it is not found, Python reads the files and Spark still builds the index.

Run:   python ir_gui.py        or        python ir_cli.py [folder]

Queries:   student                                      term
           online learning                              phrase
           online learning AND student assessment       Boolean AND
           online learning AND NOT student assessment   Boolean AND NOT
