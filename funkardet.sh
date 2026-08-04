#!/usr/bin/env sh
set -e

# Use the project virtual environment explicitly.
PYTHON=/home/stoffe/stoffe_rag_py3.14/bin/python
cd "$(dirname "$0")"

$PYTHON src/main/populate_vector_db.py --limit=1 && \
$PYTHON src/main/run.py
