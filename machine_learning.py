# ================================
# IMPORTS
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

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

# ================================
# CARREGAMENTO E TRATAMENTO
# ================================
df = pd.read_csv('flights_tratado.csv', delimiter=',')

# Amostra aleatória (evita viés)
df = df.sample(2000, random_state=42)

# Seleção de colunas
df = df[['distance', 'carrier', 'periodo_dia', 'is_delay']]

# Remoção de nulos
df = df.dropna()

# ================================
# DUMMIES
# ================================
df_dummies = pd.get_dummies(
    df,
    columns=['carrier', 'periodo_dia'],
    dtype=int,
    drop_first=True
)

# Remove colunas constantes (segurança extra)
df_dummies = df_dummies.loc[:, df_dummies.nunique() > 1]

# ================================
# TRAIN / TEST SPLIT
# ================================
X = df_dummies.drop(columns=['is_delay'])
y = df_dummies['is_delay']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# ================================
# FÓRMULA
# ================================
lista_colunas = list(X_train.columns)
formula = 'is_delay ~ ' + ' + '.join(lista_colunas)

print("\nFórmula do modelo:")
print(formula)

# ================================
# MODELO LOGÍSTICO (TREINO)
# ================================
modelo = sm.Logit.from_formula(formula, df_train).fit()

# Stepwise
modelo_step = stepwise(modelo, pvalue_limit=0.05)

print(modelo_step.summary())

# ================================
# PREDIÇÃO NO TESTE
# ================================
df_test['phat'] = modelo_step.predict(df_test)

# Probabilidade prevista
y_prob_sm = df_test['phat']

# Classificação binária
y_pred_sm = (y_prob_sm >= 0.5).astype(int)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_sm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Statsmodels - Matriz de Confusão")
plt.show()

# ================================
# CURVA ROC (TESTE)
# ================================
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

# ================================
# ODDS RATIO (INSIGHT DE NEGÓCIO)
# ================================
odds_ratios = np.exp(modelo_step.params).sort_values(ascending=False)

print("\nOdds Ratios:")
print(odds_ratios)

# ================================
# MODELO 2 - SKLEARN (REGULARIZADO)
# ================================
from sklearn.linear_model import LogisticRegression

modelo_sklearn = LogisticRegression(
    penalty='l2',
    max_iter=1000
)

modelo_sklearn.fit(X_train, y_train)

# Probabilidades no teste
y_prob_sklearn = modelo_sklearn.predict_proba(X_test)[:, 1]

# Classificação
y_pred_sklearn = (y_prob_sklearn >= 0.5).astype(int)

# MATRIZ DE CONFUSÃO
cm_sk = confusion_matrix(y_test, y_pred_sklearn)
disp_sk = ConfusionMatrixDisplay(confusion_matrix=cm_sk)
disp_sk.plot()
plt.title("Matriz de Confusão - Sklearn")
plt.show()

# ================================
# MÉTRICAS - SKLEARN
# ================================
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

fpr_sk, tpr_sk, _ = roc_curve(y_test, y_prob_sklearn)
auc_sklearn = auc(fpr_sk, tpr_sk)
gini_sklearn = 2 * auc_sklearn - 1


# ================================
# MÉTRICAS - STATSMODELS (SEU MODELO)
# ================================
y_prob_sm = df_test['phat']
y_pred_sm = (y_prob_sm >= 0.5).astype(int)

acc_sm = accuracy_score(y_test, y_pred_sm)

fpr_sm, tpr_sm, _ = roc_curve(y_test, y_prob_sm)
auc_sm = auc(fpr_sm, tpr_sm)
gini_sm = 2 * auc_sm - 1


# ================================
# COMPARAÇÃO FINAL
# ================================
comparacao = pd.DataFrame({
    'Modelo': ['Statsmodels (Stepwise)', 'Sklearn (Regularizado)'],
    'Acurácia': [acc_sm, acc_sklearn],
    'ROC AUC': [auc_sm, auc_sklearn],
    'GINI': [gini_sm, gini_sklearn]
})

# Curva ROC
print("\nComparação de Modelos:")
print(comparacao)

plt.figure(figsize=(10,6))

plt.plot(fpr_sm, tpr_sm, label=f'Statsmodels (AUC = {auc_sm:.3f})')
plt.plot(fpr_sk, tpr_sk, label=f'Sklearn (AUC = {auc_sklearn:.3f})')

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('1 - Especificidade')
plt.ylabel('Sensitividade')
plt.title('Comparação Curva ROC')
plt.legend()
plt.show()
