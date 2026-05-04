 Previsão de Atrasos em Voos com Regressão Logística
 
- Objetivo

Desenvolver um modelo preditivo para identificar a probabilidade de atraso em voos com base em características operacionais como distância, companhia aérea e período do dia.

- Metodologia

O projeto foi estruturado em etapas:

Pré-processamento
Remoção de valores nulos
Codificação de variáveis categóricas (dummies)
Remoção de variáveis constantes
Divisão dos dados
Separação treino/teste (70/30)
Estratificação da variável alvo
Modelagem
Modelo 1: Regressão Logística com statsmodels + Stepwise
Modelo 2: Regressão Logística com scikit-learn (regularização L2)
Avaliação
Acurácia
Curva ROC
AUC
Gini
Matriz de confusão

- Resultados
  
Modelo	               | Acurácia |	ROC AUC |	GINI
Statsmodels (Stepwise)	  ~0.64     	~0.63	    ~0.26
Sklearn (Regularizado)	  ~0.63	     ~0.64	    ~0.29

- Principais Insights
  
Voos no período noturno apresentam maior probabilidade de atraso
Existem diferenças relevantes entre companhias aéreas
A distância possui impacto menor comparado às variáveis categóricas

- Comparação de Modelos
  
Statsmodels
Mais interpretável (odds ratio)
Melhor para análise de negócio
Sklearn
Melhor capacidade preditiva (ROC AUC maior)
Mais robusto para produção

- Conclusão

O modelo regularizado apresentou melhor desempenho preditivo, enquanto o modelo estatístico permitiu maior interpretabilidade dos fatores de atraso.

A combinação dos dois fornece uma abordagem equilibrada entre explicação e previsão.

- Tecnologias Utilizadas
  
Python
Pandas
NumPy
Matplotlib
Statsmodels
Scikit-learn

- Próximo Passo

Testar modelos mais complexos (Random Forest, XGBoost)
