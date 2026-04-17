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
df = df.sample(1000, random_state=42) # reduzindo a amostra devido ao tamanho do dataset
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

# Estimação do modelo por meio do procedimento Stepwise para remoção de variáveis estatisticamente não significantes para 95% de confiança (p-value > 0.05)
step_modelo_is_delay = stepwise(modelo_is_delay, pvalue_limit=0.05)

# Função para a definição da matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

# Construção da matriz de confusão

# Adicionando os valores previstos de probabilidade na base de dados
df_dummies['phat'] = step_modelo_is_delay.predict()

# Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df_dummies['is_delay'],
                predicts=df_dummies['phat'],
                cutoff=0.50)
