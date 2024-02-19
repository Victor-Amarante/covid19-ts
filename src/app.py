import os
import time
import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# Definindo os diretórios
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")

def carregar_tratar_base(file_path:str , frequency:str='D') -> pd.DataFrame:
    """Carregar e fazer o tratamento da base de dados.

    Args:
        data_directory (str): Caminho do diretório onde fica a base de dados.
        frequency (str, optional): Frequência de agrupamento. Por padrão fica selecionado a frequência 'D' (Diário).

    Returns:
        pd.DataFrame: DataFrame contendo a base de dados carregada e tratada.
    """
    df = pd.read_csv(file_path, sep=";", parse_dates=["date"])
    df = df[['date','newCases']]
    # Agrupando os dados pelo tipo de frequência escolhida
    df = df[['date', 'newCases']].groupby(pd.Grouper(key= 'date', freq= frequency)).sum().reset_index()
    # Modificando o index do dataframe, formato aceito para séries temporais
    df = df.set_index('date')
    df.index.freq = frequency
    # Verificando se existem valores não positivos
    if (df['newCases'] <= 0).any():
        # Adicionando uma constante aos dados para torná-los positivos
        df['newCases'] += abs(df['newCases'].min()) + 1  # Adiciona a diferença máxima mais 1 para garantir que todos sejam positivos

    return df
    

def decomposicao_sazonal(dataframe: pd.DataFrame) -> None:
    """Realiza a decomposição sazonal dos dados.

    Args:
        dataframe (pd.DataFrame): DataFrame contendo a série temporal a ser decomposta.

    Returns:
        None
    """
    decompose_result = seasonal_decompose(dataframe['newCases'],model='multiplicative')
    graph_plot = decompose_result.plot()
    return graph_plot

def analisando_modelo(dataframe: pd.DataFrame):
    """Realiza a análise do modelo de suavização exponencial triplo de Holt-Winters.

    Args:
        dataframe (pd.DataFrame): DataFrame contendo os dados a serem analisados.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: Gráfico com a análise do modelo.
    """
    dataframe['HWES3_MUL'] = ExponentialSmoothing(dataframe['newCases'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
    plot_analise = dataframe[['newCases','HWES3_MUL']].plot(title='Holt Winters Triple Exponential Smoothing (Daily data)')
    return plot_analise


def treinando_modelo(train_df: pd.DataFrame, seasonal_periods: int, forecast: int):
    """Treina o modelo de suavização exponencial de Holt-Winters.

    Args:
        train_df (pd.DataFrame): DataFrame contendo os dados de treinamento.
        seasonal_periods (int): Número de períodos sazonais.
        forecast (int): Número de períodos a serem previstos.

    Returns:
        None
    """
    fitted_model = ExponentialSmoothing(train_df['newCases'],trend='mul',seasonal='mul',seasonal_periods=seasonal_periods).fit()
    test_predictions = fitted_model.forecast(forecast)
    test_predictions.plot(legend=True,label='PREDICTION')
    plt.title('Predicted Test using Holt Winters')
    
    
def forecast_data(dataframe: pd.DataFrame, frequency: str):
    """Faz a previsão de dados utilizando suavização exponencial de Holt-Winters.

    Args:
        dataframe (pd.DataFrame): DataFrame contendo os dados a serem previstos.
        frequency (str): Frequência dos dados ('daily' para diário, 'monthly' para mensal).

    Returns:
        None
    """
    if frequency == "daily":
        dataframe.index.freq = 'D'
        train = dataframe[:1000]
        test = dataframe[1000:]
        treinando_modelo(train, 365, 31)
    elif frequency == "monthly":
        dataframe.index.freq = 'ME'
        train = dataframe[:37]
        test = dataframe[37:]
        treinando_modelo(train, 12, 6)



df = carregar_tratar_base(file_path='../data/covid19_PE.csv', frequency='M')
df[['newCases']].plot(title='New Cases of Covid-19')
decomposicao_sazonal(dataframe=df)
analisando_modelo(df)
forecast_data(df, frequency='monthly')

