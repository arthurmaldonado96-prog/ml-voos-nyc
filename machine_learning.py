import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col

df = pd.read_csv('flights_tratado.csv', delimiter=',')
df = df.iloc[:1000]
df.info()

# Selecionamos apenas colunas relevantes para o modelo
# (variáveis preditoras + variável dependente)
df = df[[
    'distance',       # distância do voo
    'carrier',        # companhia aérea
    'periodo_dia',    # período do dia
    'is_delay'        # variável dependente (0 ou 1)
]]

# Remove linhas com valores nulos para evitar erros no modelo
df = df.dropna()

# Tabela de frequências absolutas das variáveis qualitativas
df['carrier'].value_counts().sort_index()
df['periodo_dia'].value_counts().sort_index()

# Dummizando as variáveis qualitativas
df_dummies = pd.get_dummies(df,
                            columns = ['carrier',
                                       'periodo_dia'],
                            dtype = int,
                            drop_first = True)
df_dummies = df_dummies.dropna()

# Definição da fórmula utilizada no modelo, devido ao número de elevado de dummies

lista_colunas = list(df_dummies.drop(columns=['is_delay']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = 'is_delay ~ ' + formula_dummies_modelo
print(formula_dummies_modelo)

modelo_is_delay = sm.Logit.from_formula(formula_dummies_modelo, df_dummies).fit()
modelo_is_delay.summary()
