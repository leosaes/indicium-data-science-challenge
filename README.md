# Desafio Cientista de Dados

Projeto de análise de dados e machine learning para previsão de notas do IMDB desenvolvido para o desafio de Cientista de Dados da Indicium.

## Estrutura do Projeto
```
INDICIUM/
├── desafio_indicium_imdb.csv     # Dataset original com dados dos filmes
├── data_cleaner.py               # Funções de limpeza e preprocessamento
├── eda_desafio.ipynb             # Análise Exploratória de Dados (EDA)
├── model_desafio.ipynb           # Modelagem e treinamento do ML
├── model_indicium.pkl            # Modelo treinado salvo
├── requirements.txt              # Dependências do projeto
├── README.md                     # Este arquivo
```

## Como Instalar e Executar

### Pré-requisitos:

- Python 3.8+ instalado

### 1. Instalar dependências:

   pip install -r requirements.txt

### 2. Executar o projeto

|__ Análise Exploratória de Dados

    Abrir Jupyter Notebook

    Navegar até eda_desafio.ipynb

    Executar todas as células para ver a análise exploratória e consultar a resposta das perguntas 1 e 2

|__ Feature Engineering e Treinamento do ML

    Abrir Jupyter Notebook

    Navegar até model_desafio.ipynb

    Executar todas as células para consultar as respostas 3 e 4, transformar os dados, treinar o modelo, avaliar a performance e salvar o modelo treinado
```
```
    Para utilizar o modelo, basta seguir o código a seguir, que também está presente na última célula do notebook:

    python
import pickle
import pandas as pd
import data_cleaner

with open('model_indicium.pkl','rb') as f:
  model = pickle.load(f)

new_movie = pd.DataFrame([{'Series_Title': 'The Shawshank Redemption', 
                            'Released_Year': '1994', 
                            'Certificate': 'A', 
                            'Runtime': '142 min', 
                            'Genre': 'Drama', 
                            'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.', 
                            'Meta_score': 80.0, 
                            'Director': 'Frank Darabont', 
                            'Star1': 'Tim Robbins', 
                            'Star2': 'Morgan Freeman', 
                            'Star3': 'Bob Gunton', 
                            'Star4': 'William Sadler', 
                            'No_of_Votes': 2343110, 
                            'Gross': '28,341,469'
}])


df_test = data_cleaner.clean_data(new_movie)
df_movie = data_cleaner.fix_dataframe(df_test,model['expected_columns'])
IMDB_Rate = model['pipeline'].predict(df_movie)

print(f"Nota do IMDB: {IMDB_Rate}")
```
```

    



