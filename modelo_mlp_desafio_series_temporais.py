# -*- coding: utf-8 -*-
"""
# **APLICAÇÃO MLP**

**1. Criação de dataset**
"""

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from datetime import datetime

url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.24363/dados?formato=json"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df['data'] = pd.to_datetime(df['data'], dayfirst=True)
df['valor'] = df['valor'].astype(float)

# Gráfico dos dados históricos
plt.figure(figsize=(11,3))
sns.lineplot(data=df,x='data',y='valor')
plt.title('Preço diário das ações de 01/01/2003 até 01/06/2023')
plt.xlabel('Data')
plt.ylabel('Valor da Ação (R$)')
plt.savefig('full_time_series.eps', format='eps')

"""# **2. Criação dataset dos últimos 05 anos para VALIDATION, TREINAMENTO e TESTE**"""

# criamos uma nova tabela dos útimos 05 anos
current_date = datetime.now()
five_years_ago = current_date.replace(year=current_date.year - 5)
filtered_data = df[df['data'] >= five_years_ago]

# criamos dataset dos 06 últimos periodos
last_six_periods=filtered_data.tail(6)

#criamos dataset dos ultimos 05 anos sem 06 últimos periodos
df = filtered_data.iloc[:-6]

df.set_index('data', inplace=True) #retiramos os índices

data_column = 'valor'
df[data_column] = pd.to_numeric(df[data_column], errors='coerce') # Tratamento para evitar valores indesejados
df.dropna(subset=[data_column], inplace=True)

"""# **3. Normalizamos os dados da tabela df nova**"""

#normalizamos os dados
scaler = MinMaxScaler() # a diferença entre o máximo e o mínimo é o padrão
df_scaled = scaler.fit_transform(df[[data_column]])

"""# **4. Criamos modelo de treinamento com MLP**"""

# Crearmos dados entrada
X = df_scaled[:-1]
y = df_scaled[1:]

# Vamos dividir conjuntos treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, shuffle=False)

# MLP modelo
model = MLPRegressor(hidden_layer_sizes=(90,), activation='logistic', max_iter=2000, random_state=42)

"""# **5. Treinamos modelo**"""

# treinarmos modelo
model.fit(X_train, y_train)

# previsão dos dados teste
y_pred = model.predict(X_test)

# tiramos normalização previsão e dados
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_original = scaler.inverse_transform(y_test)

# (MSE =ERRO QUADRADO MEDIO)
mse = mean_squared_error(y_test_original, y_pred)
print(f"MSE: {mse}")

"""# **5. Plotamos o modelo treinado vs dados para validation**"""

#normalizamos o dataset last_six_periods =datset de VALIDATION
scaler = MinMaxScaler()
last_six_periods_scaled = scaler.fit_transform(last_six_periods[['valor']])

# colocamos os dados dentro do modelo treinado
predicted_values = model.predict(last_six_periods_scaled)

# Desnormalizar previsões
predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1))

date_strings = ['01/01/2023', '01/02/2023', '01/03/2023', '01/04/2023', '01/05/2023', '01/06/2023']

df_predicted_values = pd.DataFrame(predicted_values, columns=['valor'])
df_predicted_values['data'] = date_strings

df_predicted_values=df_predicted_values[['data', 'valor']]

# Criar um gráfico usando matplotlib
plt.plot(df_predicted_values['data'],df_predicted_values['valor'], label='Predicted Values')
plt.plot(df_predicted_values['data'],last_six_periods['valor'], label='Real Values')

# Adicionar rótulos e título
plt.xlabel('Data')
plt.ylabel('Valor')
plt.title('Real vs. Predicted Values')

# Adicionar legenda
plt.legend()
plt.show()