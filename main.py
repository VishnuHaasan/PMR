import time
from subprocess import Popen
import json
import os

if __name__ == "__main__":

    start = time.time()

    models_to_run = [
        "lr",
        "etr",
        "dtr",
        "gbr"
    ]
    name = "pobop"
    data = "useful.csv"

    d = {
        "target": "probability"
    }
    parent = f"config/{name}"
    if not os.path.exists(parent):
        os.makedirs(parent)
    file = open(f"config/{name}/index.json", "w+")
    json.dump(d, fp=file)

    for idx, m in enumerate(models_to_run, 1):
        command = ["python3", "executor.py",  f"{m}", f"{idx}", f"{data}", f"{name}"]
        print(' '.join(command))
        Popen(command, stdout=None, stderr=None, stdin=None, close_fds=True)

    