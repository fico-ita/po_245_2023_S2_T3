# %%
import math
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bcb import Expectativas
from scipy.interpolate import interp1d

# Suprimir temporariamente os avisos
warnings.filterwarnings("ignore")


# %%
def Tratar_SELIC():
    link = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados?formato=json"

    requisicao = requests.get(link)
    informacoes = requisicao.json()

    ## Todos as vezes que existem uma lista de dicionários organizados é possível transformar em uma tabela
    tabela_anual = pd.DataFrame(informacoes)

    # Transformar a coluna 'valor' em tipo pd.float
    tabela_anual["valor"] = tabela_anual["valor"].astype(float)
    tabela_anual = tabela_anual.set_index("data")
    tabela_anual = tabela_anual.rename(columns={"valor": "anula100"})

    # 1737- IPCA - Série histórica com número-índice, variação mensal e variações acumuladas em 3 meses, em 6 meses, no ano e em 12 meses (a partir de dezembro/1979)
    link = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json"

    requisicao = requests.get(link)
    informacoes = requisicao.json()
    ## Todos as vezes que existem uma lista de dicionários organizados é possível transformar em uma tabela
    tabela_diario = pd.DataFrame(informacoes)

    # Transformar a coluna 'valor' em tipo pd.float
    tabela_diario["valor"] = tabela_diario["valor"].astype(float)
    tabela_diario = tabela_diario.set_index("data")
    tabela_diario = tabela_diario.rename(columns={"valor": "diario"})

    df_Selic = pd.concat([tabela_anual, tabela_diario], axis=1)
    ## Coluna data como índice
    df_Selic.index = pd.to_datetime(df_Selic.index, format="%d/%m/%Y")

    ## Dados da Selic
    arquivo_Selic = os.path.join("dataset", "BR", "Selic", "Selic.parquet")
    # arquivo_Selic = os.path.join("dataset", "BR","Selic", arquivo_Selic)

    # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
    df_Selic = df_Selic.replace([np.inf, -np.inf], np.nan)
    df_Selic.to_parquet(arquivo_Selic, engine="fastparquet")
    print("Fim do Tratamento de df_Selic")


# %%
def Func_Expec_Selic():
    # Importando os dados do BCB
    # Instancia a classe
    em = Expectativas()

    SerieCaptada = "ExpectativasMercadoSelic"
    ep = em.get_endpoint(SerieCaptada)

    # Dados da Selic
    selic_expec = (
        ep.query()
        .filter(ep.Indicador == "Selic")
        .filter(ep.baseCalculo == "0")
        .orderby("Data asc")
        .select(
            ep.Indicador,
            ep.Data,
            ep.Reuniao,
            ep.Media,
            ep.Mediana,
            ep.DesvioPadrao,
            ep.baseCalculo,
        )
        .collect()
    )

    def calcular_data_reuniao(reuniao):
        # Extrai o número da reunião e o ano
        numero_reuniao, ano = map(int, reuniao[1:].split("/"))

        # Calcula a data da primeira reunião do ano
        primeira_reuniao_data = datetime(
            ano,
            2,
            1,
        )  # Supondo que a primeira reunião seja em 1 de fevereiro

        # Calcula a data da reunião com base no número da reunião e a diferença de dias
        dias_reuniao = (numero_reuniao - 1) * 45
        data_reuniao = primeira_reuniao_data + timedelta(days=dias_reuniao)

        return data_reuniao

    # Exemplo de uso
    reuniao = "R2/2022"
    data = calcular_data_reuniao(reuniao)
    # print(data)

    df_filtrado2 = selic_expec.copy()
    df_filtrado2["Reuniao"] = df_filtrado2["Reuniao"].apply(calcular_data_reuniao)
    df_filtrado2["Diferenca"] = (df_filtrado2["Reuniao"] - df_filtrado2["Data"]).dt.days
    df_filtrado2 = df_filtrado2[df_filtrado2["Data"] > datetime(2000, 1, 1)]

    grupos = df_filtrado2.groupby("Data")
    dic_selic = {}
    Selic_12Meses = pd.DataFrame(columns=["Valor"])
    for data, grupo in grupos:
        # print(data, grupo)

        df_temp = grupo.loc[:, ["Data", "Mediana", "Diferenca"]]

        # Crie uma função de interpolação usando interp1d
        interpolacao = interp1d(
            df_temp["Diferenca"],
            df_temp["Mediana"],
            fill_value="extrapolate",
        )
        valor_interpolacao = np.round(interpolacao(365), 2)
        df_temp.loc[df_temp.index[-1] + 1, :] = [data, valor_interpolacao, 365]
        dic_selic[data] = df_temp
        Selic_12Meses.loc[data, "Valor"] = valor_interpolacao

    Selic_12Meses = Selic_12Meses.dropna()

    ## Colunas do ARIMA
    # for i in range(1,13):
    #   Selic_12Meses[f"{i}_mes"] = 0

    return Selic_12Meses


# %%
# Defina uma função que calcula o número de indicações de alta entre todas as combinações
def calcular_indicacoes_alta(row):
    indicacoes = 0
    colunas = row.loc["MM1":"MM12"].index
    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            if row[colunas[i]] > row[colunas[j]]:
                indicacoes += 1

    # Montando a combinação
    n = len(colunas)
    k = 2
    comb_possiveis = math.factorial(n) / (k * (math.factorial(n - k)))
    indicacoes_percentual = indicacoes / comb_possiveis
    return indicacoes_percentual


# %%
def Tratar_expec_SELIC():
    Selic_12Meses = Func_Expec_Selic()

    Selic_12Meses["MM1"] = (
        Selic_12Meses["Valor"].rolling(window=21, min_periods=21).mean()
    )

    Selic_12Meses["MM3"] = (
        Selic_12Meses["Valor"].rolling(window=21 * 3, min_periods=21).mean()
    )

    Selic_12Meses["MM6"] = (
        Selic_12Meses["Valor"].rolling(window=21 * 6, min_periods=21).mean()
    )

    Selic_12Meses["MM9"] = (
        Selic_12Meses["Valor"].rolling(window=21 * 9, min_periods=21).mean()
    )

    Selic_12Meses["MM12"] = (
        Selic_12Meses["Valor"].rolling(window=21 * 12, min_periods=21).mean()
    )

    Selic_12Meses = Selic_12Meses.dropna()

    # Aplique a função a cada linha do DataFrame e crie uma nova coluna
    Selic_12Meses["Indicacoes_Alta"] = Selic_12Meses.apply(
        calcular_indicacoes_alta,
        axis=1,
    )

    nome_arquivo = "Expectativa_Selic_Diaria.parquet"
    arquivo_Selic = os.path.join("dataset", "BR", "Selic", nome_arquivo)
    # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
    Selic_12Meses = Selic_12Meses.replace([np.inf, -np.inf], np.nan)
    Selic_12Meses.to_parquet(arquivo_Selic, engine="fastparquet")
    print("Fim do Tratamento de Expectativa Selic")


# %%
def Tratar_baixar_SELIC_expec_SELIC():
    Tratar_SELIC()
    Tratar_expec_SELIC()
