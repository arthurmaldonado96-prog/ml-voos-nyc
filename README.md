## 📌 Objetivo

Desenvolver um modelo de Machine Learning capaz de prever a probabilidade de atraso em voos, com base em variáveis operacionais como atraso na decolagem, características do voo e período do dia.

## Dataset

O projeto utiliza o dataset **NYC Flights 2013**, contendo informações sobre voos domésticos, incluindo horários, atrasos, companhias aéreas e aeroportos.

## Tecnologias Utilizadas

* Python
* Pandas
* Scikit-learn


## Pipeline do Projeto

### 1. Preparação dos Dados

* Seleção de variáveis relevantes
* Remoção de valores nulos
* Dummização de variáveis categóricas

### 2. Engenharia de Features

* Criação da variável dependente `is_delay`
* Criação da variável preditora `periodo_dia`

### 3. Modelagem

Foram utilizados dois modelos:

* Regressão Logística
* Random Forest


## Divisão dos Dados

Os dados foram divididos em:

* 80% para treino
* 20% para teste

Utilizando `train_test_split` para garantir avaliação em dados não vistos.


## Avaliação dos Modelos

As seguintes métricas foram utilizadas:

* Accuracy
* Precision
* Recall
* F1-score

A partir da matriz de confusão.



## 👨‍💻 Autor

Arthur M

