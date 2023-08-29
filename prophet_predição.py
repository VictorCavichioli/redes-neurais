# -*- coding: utf-8 -*-
"""
# **APLICAÇÃO PROPHET**

**1. Criação de dataset**
"""

import pandas as pd
from prophet import Prophet
import requests
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.24363/dados?formato=json"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df['data'] = pd.to_datetime(df['data'], dayfirst=True)
df['valor'] = df['valor'].astype(float)

# Renomeia as colunas para "ds" (data) e "y" (valor)
df.columns = ["ds", "y"]

# Criação do modelo Prophet
model = Prophet()

# Treinamento do modelo
model.fit(df)

# Criação do DataFrame de datas futuras
future = model.make_future_dataframe(periods=6, freq="MS")

# Realiza as previsões
forecast = model.predict(future)

# Plot dos resultados históricos e de previsão
fig = model.plot(forecast)
plt.title("Índice de Atividade Econômica - Previsão")
plt.xlabel("Data")
plt.ylabel("Índice")
plt.show()

# Plot dos componentes do modelo
fig_comp = model.plot_components(forecast)
plt.show()

from sklearn.metrics import mean_squared_error

# Valores reais
y_true = df["y"]

# Valores previstos
y_pred = forecast["yhat"].tail(len(y_true))

# Cálculo do MSE
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", mse)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Cálculo do MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print("Mean Absolute Percentage Error:", mape)

direction_accuracy = (sum((y_true.shift(1) < y_true) & (y_pred.shift(1) < y_pred)) +
                      sum((y_true.shift(1) > y_true) & (y_pred.shift(1) > y_pred))) / len(y_true)
print("Directional Accuracy:", direction_accuracy)