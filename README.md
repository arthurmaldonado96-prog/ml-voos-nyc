## Objetivo

Desenvolver um modelo de Machine Learning capaz de prever a probabilidade de atraso em voos, com base em variáveis operacionais como atraso na decolagem, características do voo e período do dia.

## Dataset

O projeto utiliza o dataset **NYC Flights 2013**, contendo informações sobre voos domésticos, incluindo horários, atrasos, companhias aéreas e aeroportos.

## Tecnologias Utilizadas

* Python
* Pandas
* Scikit-learn



## Preparação dos Dados

* Seleção de variáveis relevantes
* Remoção de valores nulos
* Dummização de variáveis categóricas

## Engenharia de Features

* Criação da variável dependente `is_delay`
* Criação da variável preditora `periodo_dia`

## Modelagem

Foi utilizado o modelo de regressão logística binária


## Avaliação 

As seguintes métricas foram utilizadas, a partir da matriz de confusão:

* Sensitividade
* Especificidade
* Acurácia
* Curva ROC

## Insinghts

A partir da análise de Odds Ratio, nota-se que os atrasos estão fortemente associados à companhia aérea e ao período do dia. Voos noturnos apresentam mais que o dobro de chance de atraso, sugerindo um efeito acumulado operacional. Além disso, há diferenças relevantes entre companhias, com algumas apresentando risco até 5 vezes maior.

## Autor

Arthur M

