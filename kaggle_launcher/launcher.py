import subprocess, sys, os

REPO = "https://github.com/P00l000/kaggle_playground_11-26.git"
DEST = "/tmp/repo"          # /tmp so the repo isn't included in the output
NOTEBOOK = "src/train.ipynb"    # path of your notebook inside the repo

subprocess.run(["git", "clone", "--depth", "1", REPO, DEST], check=True)


if os.path.exists(f"{DEST}/requirements.txt"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{DEST}/requirements.txt"], check=True)

os.chdir(DEST)
env = {**os.environ, "PYTHONPATH": DEST}
subprocess.run([
    sys.executable, "-m", "jupyter","ls", "-R", "/kaggle/input", "nbconvert",
    "--to", "notebook", "--execute", NOTEBOOK,
    "--output-dir", "/kaggle/working",
    "--output", "result.ipynb",
    "--ExecutePreprocessor.timeout=-1",   # important, see below
], check=True)