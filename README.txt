PySpark Information Retrieval application
=========================================
Files
  ir_engine.py  core: dataset import, Positional Index (PySpark), TF/DF/IDF/TF-IDF,
                normalization, query processing, similarity, ranking
  ir_cli.py     command-line application
  ir_gui.py     graphical interface (bonus)
  datasets/education            the assigned dataset (d1.txt ... d10.txt)
  datasets/sample_other_domain  a different dataset (7 files, other names) to show flexibility

Requirements: Python 3.10+, Java 17+, pip install pyspark

Run (command line)
  python ir_cli.py                       -> asks for the dataset folder, then shows a menu
  python ir_cli.py datasets\education    -> uses this folder
  python ir_cli.py datasets\education --all
  python ir_cli.py datasets\education --query "online learning AND NOT student assessment"

Run (GUI)
  python ir_gui.py

Query syntax
  student                                   term search
  online learning                           phrase query
  online learning AND student assessment    Boolean AND
  online learning AND NOT student assessment  Boolean AND NOT

Letting Spark read the files itself (Windows)
  The application asks Spark (wholeTextFiles) to read the dataset folder. On Windows Spark needs
  winutils.exe and hadoop.dll. The program looks for a folder named hadoop-3.x that contains
  bin\winutils.exe in these places: HADOOP_HOME, ir_app\hadoop, C:\hadoop, C:\winutils and
  your Downloads / Desktop / Documents folders (for example Downloads\winutils-master\hadoop-3.0.0).
  If none is found, it reads the files with Python and still builds the index with Spark;
  the Dataset tab / status bar tells you which way was used.
  To enable it manually: copy hadoop-3.0.0 (the folder that contains bin\winutils.exe and
  bin\hadoop.dll) to ir_app\hadoop  and run again.
