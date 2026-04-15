import pandas as pd

df = pd.read_csv('nyc_flights.csv', sep=',')

# removendo voos sem informação de atraso
df = df.dropna(subset=['dep_delay', 'arr_delay'])

# criando variável de atraso
df['is_delay'] = df['arr_delay'].apply(lambda x: 1 if x > 0 else 0)

# criando coluna de datas
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# período do dia
def periodo(h):
    if pd.isna(h):
        return 'Desconhecido'
    elif 5 <= h < 12:
        return 'Manhã'
    elif 12 <= h < 18:
        return 'Tarde'
    else:
        return 'Noite'
    
df['periodo_dia'] = df['dep_time'].apply(lambda x: periodo(int(x/100)))
    
# salvando
df.to_csv('nyc_flights.csv', index = False)

print('ETL OK')