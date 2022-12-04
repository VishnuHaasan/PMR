import time
from subprocess import Popen
import json
import os
import pickle
import sys
import shutil
import pandas as pd

class PMR:
    def __init__(self, n_jobs) -> None:
        self.n_jobs = n_jobs

    def __saveConfig(self) -> bool:
        try:
            self.parent = f"config/{self.name}"
            d = self.__dict__
            if not os.path.exists(self.parent):
                os.makedirs(self.parent)
            file = open(f"{self.parent}/index.json", "w")
            json.dump(d, fp=file, indent=4)
            file.close()
            return True
        except Exception as e:
            print(e)
            return False
            

    def fit(self, config, dataframe : pd.DataFrame):
        self.target = config["target"]
        self.cv = config["cv"]
        self.name = config["name"]
        self.datapath =  f"data/{self.name}.csv"
        dataframe.to_csv(path_or_buf = self.datapath,index=False)
        return self.__saveConfig()

    def __batchDelete(self):
        try:
            conf = f"config/{self.name}"
            pick = f"pickles/{self.name}"
            shutil.rmtree(conf)
            shutil.rmtree(pick)
            os.remove(self.datapath)
        except Exception as e:
            print(e)

    def __updateMainFile(self, model_data):
        file = open(f"{self.parent}/index.json", "r")
        d = json.load(fp=file)
        d['model_info'] = model_data
        file.close()
        return d

    def __extractModel(self, model_index):
        pick = f"pickles/{self.name}/model{model_index}.pickle"
        file = open(pick, "rb")
        obj = pickle.load(file=file)
        file.close()
        return obj

        
    def transform(self):
        start = time.time()

        models_to_run = [
            "lr",
            "etr",
            "dtr",
            "gbr"
        ]

        mTotal = 1

        while(mTotal <= len(models_to_run)):
            for i in range(self.n_jobs):
                if mTotal > len(models_to_run):
                    break
                command = ["python3", "executor.py", f"{mTotal}", f"{self.parent}/index.json", f"{models_to_run[mTotal-1]}"]
                Popen(command, stderr=None, stdout=None, stdin=None, close_fds=True)
                mTotal += 1
            while(len(os.listdir(path=self.parent)) < mTotal):
                continue

        min_score = sys.maxsize
        model_index = -1
        model_data = {}
        
        for i in range(1, len(models_to_run) + 1):
            file = open(f"{self.parent}/conf{i}.json")
            d = json.load(fp=file)
            if self.cv:
                score = d['cv_score']
            else:
                score = d['mse']
            if score < min_score:
                min_score = score
                model_index = i
                model_data = d
            file.close()

        main_data =  self.__updateMainFile(model_data=model_data)

        model = self.__extractModel(model_index=model_index)

        self.__batchDelete()

        elapsed = time.time() - start

        main_data['elapsed_time'] = elapsed

        return main_data, model







    