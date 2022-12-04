from pmr import PMR
import pandas as pd


if __name__ == "__main__":
    newPmr = PMR(n_jobs=4)
    df = pd.read_csv("data/useful.csv")
    config = {
        "target": "probability",
        "cv": True,
        "name": "fourrun"
    }
    newPmr.fit(config=config, dataframe = df)
    data, model = newPmr.transform()
    print(data)
    print(model.__class__.__name__)