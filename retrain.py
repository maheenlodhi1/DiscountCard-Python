#!/usr/bin/env python3
import subprocess
import sys
import os

def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    py = sys.executable  # use the same interpreter as your venv

    print("[Retrain] Recommender (mongo)…")
    run([py, "train_recommend.py", "--mongo"])

    print("[Retrain] Trending (mongo)…")
    run([py, "train_trending.py", "--mongo"])

    print("[Retrain] Completed.")

if __name__ == "__main__":
    main()
