from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import sys
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import os

lr = LinearRegression()

etr = ExtraTreesRegressor(random_state=10)

gbr = GradientBoostingRegressor(random_state=10)

dtr = DecisionTreeRegressor(random_state=10)

model_dict = {
    'lr': lr,
    'etr': etr,
    'gbr': gbr,
    'dtr': dtr
}

def train_loss(m, train, target):
    try:
        train_x, test_x, train_y, test_y = train_test_split(train, target, train_size=0.8,random_state=10)
        m.fit(train_x, train_y)
        pred = m.predict(test_x)
        mse = mean_squared_error(y_true=test_y, y_pred=pred)
        pred = m.predict(train)
        mape = mean_absolute_percentage_error(y_true=target, y_pred=pred)
        return {
            "mse": mse,
            "mape": mape,
            "status": "S"
        }
    except Exception as e:
        print(e)
        return {
            "status": "E",
            "err": e
        }

def write_file(p, data):
    f = open(f"config/{p}", "w+")
    json.dump(data, f)

def save_pickle(p, m):
    f = open(f"pickles/{p}", "wb+")
    pickle.dump(m, f)

def execute(model, path, ind, name):
    try:
        df = pd.read_csv(f"data/{path}")
        print(df.head())
        m = model_dict[model]
        file = open(f"config/{name}/index.json")
        conf = json.load(file)
        target = df[conf['target']]
        print(conf['target'])
        train_df = df.drop(columns=[conf['target']])
        train_res = train_loss(m=m, train=train_df, target=target)
        pickle_name = f"model{ind}.pickle"
        conf_name = f"conf{ind}.json"
        pickle_path = f"pickles/{name}/{pickle_name}"
        conf_path = f"conf/{name}/{conf_name}"
        save_pickle(pickle_name, m)
        write_file(conf_name, train_res)
        return True
    except Exception as e:
        print(e)
        return False

def init(name):
    curr = os.getcwd()
    p = f"{curr}/pickles/{name}"
    if not os.path.exists(p):
        os.makedirs(p)
    p = f"{curr}/config/{name}"
    if not os.path.exists(p):
        os.makedirs(p)

    

    

if __name__ == "__main__":
    args = sys.argv[1:]
    model = args[0]
    ind = args[1]
    path = args[2]
    name = args[3]
    init(name=name)
    if execute(model, path, ind, name):
        print("Success")
    else:
        print("Failure")

    




