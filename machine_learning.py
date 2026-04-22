
# IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    recall_score,
    roc_curve,
    auc
)

from statstests.process import stepwise


# CARREGAMENTO E TRATAMENTO

df = pd.read_csv('flights_tratado.csv', delimiter=',')

# Amostra aleatória (evita viés)
df = df.sample(2000, random_state=42)

# Seleção de colunas
df = df[['distance', 'carrier', 'periodo_dia', 'is_delay']]

# Remoção de nulos
df = df.dropna()

# DUMMIZAÇÃO DAS VARIÁVEIS CATEGÓRICAS

df_dummies = pd.get_dummies(
    df,
    columns=['carrier', 'periodo_dia'],
    dtype=int,
    drop_first=True
)

# Remove colunas constantes (segurança extra)
df_dummies = df_dummies.loc[:, df_dummies.nunique() > 1]

# TRAIN / TEST SPLIT

X = df_dummies.drop(columns=['is_delay'])
y = df_dummies['is_delay']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# FÓRMULA

lista_colunas = list(X_train.columns)
formula = 'is_delay ~ ' + ' + '.join(lista_colunas)

print("\nFórmula do modelo:")
print(formula)

# MODELO LOGÍSTICO (TREINO)

modelo = sm.Logit.from_formula(formula, df_train).fit()

# Algoritmo Stepwise para remover variáveis estatisticamente não significantes 
modelo_step = stepwise(modelo, pvalue_limit=0.05)

print(modelo_step.summary())

# PREDIÇÃO NO TESTE

df_test['phat'] = modelo_step.predict(df_test)


# MATRIZ DE CONFUSÃO

def matriz_confusao(predicts, observado, cutoff):

    predicao_binaria = (predicts >= cutoff).astype(int)

    # ORDEM CORRETA
    cm = confusion_matrix(observado, predicao_binaria)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()

    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    indicadores = pd.DataFrame({
        'Sensitividade': [sensitividade],
        'Especificidade': [especificidade],
        'Acurácia': [acuracia]
    })

    return indicadores


print("\nMatriz de Confusão:")
print(
    matriz_confusao(
        observado=df_test['is_delay'],
        predicts=df_test['phat'],
        cutoff=0.5
    )
)


# CURVA ROC (TESTE)

fpr, tpr, thresholds = roc_curve(
    df_test['is_delay'],
    df_test['phat']
)

roc_auc = auc(fpr, tpr)
gini = 2 * roc_auc - 1

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, marker='o', label=f'AUC = {roc_auc:.3f}')
plt.plot(fpr, fpr, linestyle='--')
plt.title(f'Curva ROC | GINI = {gini:.3f}')
plt.xlabel('1 - Especificidade')
plt.ylabel('Sensitividade')
plt.legend()
plt.show()


# ODDS RATIO

odds_ratios = np.exp(modelo_step.params).sort_values(ascending=False)

print("\nOdds Ratios:")
print(odds_ratios)
