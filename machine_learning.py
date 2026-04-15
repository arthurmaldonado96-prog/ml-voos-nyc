# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# IMPORTAÇÃO DE BIBLIOTECAS

import pandas as pd  # manipulação de dados
from sklearn.model_selection import train_test_split  # divisão treino/teste
from sklearn.linear_model import LogisticRegression  # modelo 1
from sklearn.ensemble import RandomForestClassifier  # modelo 2
from sklearn.metrics import classification_report, confusion_matrix  # avaliação


# 1. CARREGAMENTO DOS DADOS

# Lê o dataset já tratado pelo ETL
df = pd.read_csv('nyc_flights.csv')

# Exibe as primeiras linhas para validação
print(df.head())


# 2. SELEÇÃO DE VARIÁVEIS

# Selecionamos apenas colunas relevantes para o modelo
# (variáveis preditoras + variável dependente)
df = df[[
    'dep_delay',      # atraso na saída
    'distance',       # distância do voo
    'carrier',        # companhia aérea
    'origin',         # aeroporto de origem
    'dest',           # aeroporto de destino
    'periodo_dia',    # período do dia
    'is_delay'        # variável dependente (0 ou 1)
]]

# Remove linhas com valores nulos para evitar erros no modelo
df = df.dropna()


# 3. TRATAMENTO DE VARIÁVEIS CATEGÓRICAS

# Processo de dummização de variáveis categóricas
df = pd.get_dummies(
    df,
    columns=['carrier', 'origin', 'dest', 'periodo_dia'],
    drop_first=True  # evita multicolinearidade
)


# 4. SEPARAÇÃO ENTRE VARIÁVEIS 

# X = variáveis preditoras (entrada do modelo)
X = df.drop('is_delay', axis=1)

# y = variável que queremos prever
y = df['is_delay']


# 5. DIVISÃO TREINO E TESTE

# Divide os dados:
# - 80% para treino
# - 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% teste
    random_state=42     # reprodutibilidade
)


# 6. TREINAMENTO DOS MODELOS

# Modelo 1: Regressão Logística 
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Modelo 2: Random Forest 
rf_model = RandomForestClassifier(
    n_estimators=100,   # número de árvores
    random_state=42
)
rf_model.fit(X_train, y_train)


# 7. AVALIAÇÃO DOS MODELOS

# Fazendo previsões com cada modelo
y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print("\n=== Logistic Regression ===")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("\n=== Random Forest ===")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))



