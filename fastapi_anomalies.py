from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.rod import ROD
from pyod.models.abod import ABOD
from pyod.models.sos import SOS
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.utils.data import generate_data
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization, majority_vote
from sklearn.preprocessing import StandardScaler
import pickle
import json


def anomalies_by_one_metric(metric, data, column, start_date, end_date, resample_interval, file_name):
    df = data.resample(resample_interval, on='ds').mean().reset_index()

    model = Prophet()
    model.fit(df)

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    actuals = df[(df['ds'] >= start_datetime) & (df['ds'] <= end_datetime)]

    future = pd.DataFrame({'ds': actuals['ds']})

    forecast = model.predict(future)

    results = pd.merge(actuals, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')

    results['anomaly'] = (results['y'] < results['yhat_lower']) | (results['y'] > results['yhat_upper'])

    return results


def plot_anomalies(results, metric_name, metric, file_name):
    plt.figure(figsize=(12, 6))
    plt.plot(results['ds'], results[metric_name], label=f'Метрика {metric}')
    plt.fill_between(results['ds'], results['yhat_lower'], results['yhat_upper'], color='red', alpha=0.3)

    anomalies = results[results['anomaly']]
    for i in range(len(anomalies)):
        if anomalies[metric_name].iloc[i] > results['yhat_upper'][i]:
            score = 1 if abs(anomalies[metric_name].iloc[i] - results['yhat_upper'][i]
                             ) > 1.5*(results['yhat_upper'][i]-results['yhat_lower'][i]) else anomalies['scores'].iloc[i]
        else:
            score = 1 if abs(anomalies[metric_name].iloc[i] - results['yhat_lower'][i]
                             ) > 1.5*(results['yhat_upper'][i]-results['yhat_lower'][i]) else anomalies['scores'].iloc[i]
        plt.scatter(anomalies['ds'].iloc[i], anomalies[metric_name].iloc[i], color='red', alpha=score, s=5+10*score)

    plt.title(f"{metric}")
    plt.legend()
    plt.savefig(f'{file_name}.png')

    results.to_csv(f'{file_name}.csv', index=False)


errors = pd.read_csv("/app/data/errors.csv")
errors.set_index("point", inplace=True)
errors = pd.DataFrame({'ds': errors.index, 'y': errors['Error_rate']})
errors['ds'] = errors['ds'].astype('datetime64[ns]')

thr = pd.read_csv("/app/data/throughput.csv")
thr.set_index("point", inplace=True)
thr = pd.DataFrame({'ds': thr.index, 'y': thr['throughput']})
thr['ds'] = thr['ds'].astype('datetime64[ns]')

webresp = pd.read_csv("/app/data/web_response.csv")
webresp.set_index("point", inplace=True)
webresp = pd.DataFrame({'ds': webresp.index, 'y': webresp['web_response_time']})
webresp['ds'] = webresp['ds'].astype('datetime64[ns]')

apdex = pd.read_csv("/app/data/apdex.csv")
apdex.set_index("point", inplace=True)
apdex = pd.DataFrame({'ds': apdex.index, 'y': apdex['APDEX']})
apdex['ds'] = apdex['ds'].astype('datetime64[ns]')


metrics_name = ["Web response", "Apdex", "Errors", "Throughput"]
metrics = [webresp, apdex, errors, thr]
metrics_filenames = ["web_response_time", 'APDEX', 'Error_rate', 'throughput']
metrics_agg = ['15T', '15T', '15T', 'H']

algos = {
    'PCA': PCA(),
    'LOF': LOF(),
    'CBLOF': CBLOF(),
    'KNN': KNN(),
    'IForest': IForest(),
    'LODA': LODA(),
}

for name in algos.keys():
    with open(f'/app/data/{name}_model.pkl', 'rb') as f:
        algos[name] = pickle.load(f)

with open('/app/data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


def scores(X_train):

    scores_pred = np.zeros([len(X_train), len(algos)])

    for i, (clf_name, clf) in enumerate(algos.items()):
        scores_pred[:, i] = clf.decision_function(X_train)

    y_by_maximization = np.sum(scores_pred)

    return y_by_maximization


def normalize_scores(scores, min_score, max_score):
    return scores/(max_score+min_score/10)


def collect_anomalies(start_date, end_date):
    max_proba = -1
    proba_time = 0

    for i in range(len(metrics)):
        result = anomalies_by_one_metric(
            metrics_name[i],
            metrics[i],
            metrics_filenames[i],
            start_date, end_date, metrics_agg[i],
            metrics_filenames[i]).rename(
            columns={'y': metrics_filenames[i]})
        anomaly = (result[result['anomaly']])[['ds', metrics_filenames[i]]]
        for j in range(len(metrics)):
            if i != j:
                anomaly = anomaly.merge(metrics[j].rename(columns={'y': metrics_filenames[j]})[
                                        ['ds', metrics_filenames[j]]], how='left', on='ds')
        anomaly = anomaly.ffill().bfill()
        anomalyd = anomaly[["APDEX", "Error_rate", "throughput", "web_response_time"]]
        anomalyd['scores'] = anomalyd.apply(lambda row: scores(row.to_numpy().reshape(1, -1)), axis=1)
        anomalyd['scores'] = normalize_scores(
            anomalyd['scores'],
            np.min(anomalyd['scores']),
            np.max(anomalyd['scores']))
        anomaly['scores'] = anomalyd['scores']
        result = result.merge(anomaly[['ds', 'scores']], how='left', on='ds').fillna(0)
        plot_anomalies(result, metrics_filenames[i], metrics_name[i], metrics_filenames[i])
        if max(anomaly['scores']) > max_proba:
            max_proba = max(anomaly['scores'])
            proba_time = anomaly[anomaly['scores'] == max_proba]['ds']
    return max_proba, proba_time


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/anomalies")
async def get_anomalies(start: str, end: str):
    start_date = datetime.fromisoformat(start)
    end_date = datetime.fromisoformat(end)

    min_date = datetime.fromisoformat("2024-04-15T23:40")
    max_date = datetime.fromisoformat("2024-05-16T00:40")
    if start_date >= end_date:
        return JSONResponse(
            status_code=400, content={"message": "Дата начала промежутка должна быть раньше, чем дата конца"})
    if start_date < min_date or end_date > max_date or start_date is None or end_date is None:
        return JSONResponse(
            status_code=400, content={"message": "Возможный промежуток: с 2024-04-15 23:40 до 2024-05-16 00:40"})
    max_proba, proba_time = collect_anomalies(start_date, end_date)
    images = []

    for name in metrics_filenames:

        with open(f'{name}.png', 'rb') as image_file:
            buf = io.BytesIO(image_file.read())
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        images.append(image_base64)

    return JSONResponse(
        content={"images": images, "max_proba": max_proba,
                 "proba_time": str(proba_time.iloc[0].isoformat())})
