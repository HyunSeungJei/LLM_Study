#!/bin/bash
set -e

# ./scripts/update_db.sh

echo "Step 1: Run LangGraph workflow"
python ../workflow/main.py

echo "Hugging Face deployed with latest DB!"

