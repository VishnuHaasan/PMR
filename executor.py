from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import sys
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import os
import copy
import numpy as np

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
        mape = mean_absolute_percentage_error(y_true=target, y_pred=pred)*100
        return {
            "mse": mse,
            "mape": mape,
            "status": "S"
        }, m
    except Exception as e:
        print(e)
        return {
            "status": "E",
            "err": e
        }, None

def train_loss_cv(m, train, target):
    try:
        mm = copy.deepcopy(m)
        res, model = train_loss(m, train=train, target=target)
        if res['status'] == "E":
            return res
        cross = cross_val_score(estimator=mm, X=train, y=target, cv=5, scoring='neg_mean_squared_error')
        res['cv_score'] = np.mean(np.sqrt(cross * -1))
        return res, model
        
    except Exception as e:
        print(e)
        return {
            "status": "E",
            "err": e
        }

def write_file(p, data):
    f = open(p, "w+")
    json.dump(data, f, indent=4)

def save_pickle(p, m):
    f = open(p, "wb+")
    pickle.dump(m, f)

def execute(model, path, ind, name, target, cv):
    try:
        df = pd.read_csv(f"{path}")
        m = model_dict[model]
        train_df = df.drop(columns=[target])
        if cv:
            train_res, m = train_loss_cv(m=m, train=train_df, target=df[target])
        else:
            train_res, m = train_loss(m=m, train=train_df, target=df[target])
        train_res['model'] = model
        train_res['model_name'] = m.__class__.__name__
        pickle_name = f"model{ind}.pickle"
        conf_name = f"conf{ind}.json"
        pickle_path = f"pickles/{name}/{pickle_name}"
        conf_path = f"config/{name}/{conf_name}"
        save_pickle(pickle_path, m)
        write_file(conf_path, train_res)
        return True
    except Exception as e:
        print(e)
        return False

def init(name):
    try:
        curr = os.getcwd()
        p = f"{curr}/pickles/{name}"
        if not os.path.exists(p):
            os.makedirs(p)
        # p = f"{curr}/config/{name}"
        # if not os.path.exists(p):
        #     os.makedirs(p)
    except Exception as e:
        print(e) 

    

    

if __name__ == "__main__":
    args = sys.argv[1:]
    ind = args[0]
    path = args[1]
    model = args[2]
    file = open(path, "r")
    conf = json.load(fp=file)
    init(name=conf['name'])
    execute(model, conf["datapath"], ind, conf["name"], conf["target"], conf["cv"])

    




