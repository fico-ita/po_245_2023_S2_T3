import os
import warnings
from datetime import datetime

import pandas as pd

# Suprimir temporariamente os avisos
warnings.filterwarnings("ignore")


def Func_ler_dados_acoes(Ticker, data_inicial, data_simulacao):
    ## Ler os dados normalizados
    arquivo_por_acao = os.path.join(
        "dataset",
        "BR",
        "ACOES",
        "Dados_Tratados",
        "Dados_Normalizados",
        "Dados_normalizados_acao_" + Ticker + ".parquet",
    )
    df_acao = pd.read_parquet(arquivo_por_acao)
    ## Filtrar a data
    condicao_selecao_data = (df_acao["Data_balanco"] <= data_simulacao) & (
        df_acao["Data_balanco"] >= data_inicial
    )
    df_acao = df_acao.loc[condicao_selecao_data, :]

    ## Ler os dados dos múltiplos
    arquivo_por_multiplos = os.path.join(
        "dataset",
        "BR",
        "ACOES",
        "Dados_Tratados",
        "Dados_Projetados",
        "Multiplos_diarios_" + Ticker + ".parquet",
    )
    df_multiplos = pd.read_parquet(arquivo_por_multiplos)
    ## Filtrar a data
    condicao_selecao_data = (df_multiplos.index <= data_simulacao) & (
        df_multiplos.index >= data_inicial
    )
    df_multiplos = df_multiplos.loc[condicao_selecao_data, :]

    ## Ler o CAGR
    arquivo_por_CAGR = os.path.join(
        "dataset",
        "BR",
        "ACOES",
        "Dados_Tratados",
        "Dados_Projetados",
        "CAGR_" + Ticker + ".parquet",
    )
    df_CAGR = pd.read_parquet(arquivo_por_CAGR)
    ## Filtrar a data
    condicao_selecao_data = (df_CAGR.index <= data_simulacao) & (
        df_CAGR.index >= data_inicial
    )
    df_CAGR = df_CAGR.loc[condicao_selecao_data, :]

    return df_acao, df_multiplos, df_CAGR


def Func_ativos_elegiveis(data_referencia):
    # Define o caminho do arquivo .parquet que contém os dados mensais do IBOVESPA
    endereco_IBOV = os.path.join("dataset", "BR", "ACOES", "ibov_comp.parquet")

    # Lê os dados do arquivo .parquet e os armazena no DataFrame 'IBOV'
    IBOV = pd.read_parquet(endereco_IBOV)

    # Converte a coluna de datas para o formato datetime
    IBOV.loc[:, "date"] = pd.to_datetime(IBOV["date"], format="%d/%m/%Y")

    # Obtém a lista de todas as datas presentes no DataFrame 'IBOV'
    todas_datas_IBOV = IBOV["date"].unique()
    todas_datas_IBOV = sorted(todas_datas_IBOV)

    # Filtra as datas que são menores que a data de referência

    datas_anteriores = [data for data in todas_datas_IBOV if data < data_referencia]

    # Obtém a maior data entre as datas filtradas
    maior_data_anterior = max(datas_anteriores)

    # Novo DataFrame com os dados do IBOV na maior data anterior à data de referência
    IBOV_new = IBOV[IBOV["date"] == maior_data_anterior]
    lista_ativos_elegiveis = list(IBOV_new["ticker"])
    lista_ativos_elegiveis.sort()

    # Todos os ativos do IBOV
    todos_ativos_IBOV = list(IBOV["ticker"].unique())
    todos_ativos_IBOV.sort()

    return lista_ativos_elegiveis, todos_ativos_IBOV


def load_data(data_simulacao, dados_portfolio=False):
    ## Atribuir uma data inicial para a simulação, alguns dados antigos estão em desacordo
    data_inicial = datetime(2000, 1, 1)

    ## Pegar os ativos elegíveis
    arquivo_IBrA = os.path.join("dataset", "BR", "ACOES", "IBrA.parquet")
    df_Elegivel = pd.read_parquet(arquivo_IBrA)
    lista_ativos_elegiveis_IBrA = list(df_Elegivel.index)

    ## Ativos Elegíveis no IBOV nesse dia
    lista_ativos_elegiveis_IBOV, todos_ativos_IBOV = Func_ativos_elegiveis(
        data_simulacao,
    )

    ## Ativos elegíveis Total (IBrA + IBOV)
    lista_ativos_elegiveis_total = list(
        set(lista_ativos_elegiveis_IBrA + todos_ativos_IBOV),
    )
    lista_ativos_elegiveis_total.sort()
    ## Contrução de Portfolio
    if dados_portfolio:
        lista_ativos_elegiveis = lista_ativos_elegiveis_IBOV
    else:  ## Benchmark
        lista_ativos_elegiveis = lista_ativos_elegiveis_total

    ## Dados da Selic
    arquivo_Selic = os.path.join("dataset", "BR", "Selic", "Selic.parquet")

    df_Selic = pd.read_parquet(arquivo_Selic)
    # Coluna data como índice
    df_Selic.index = pd.to_datetime(df_Selic.index, format="%d/%m/%Y")

    # Filtrar a data de simulação
    condicao_selecao_data = (df_Selic.index <= data_simulacao) & (
        df_Selic.index >= data_inicial
    )
    df_Selic = df_Selic.loc[condicao_selecao_data, :]

    ## Criar um ativo chamado LFTS3, representando a Selic
    df_Selic.loc[:, "LFTS3"] = df_Selic.loc[:, "diario"] / 100 + 1
    df_Selic.loc[:, "LFTS3"] = df_Selic.loc[:, "LFTS3"].cumprod()

    ## (FIM) Dados da Selic
    ## Dados do IBOV (índice Bovespa)
    arq_IBOV = "Cotacao_IBOV.parquet"
    ibov = os.path.join(
        "dataset",
        "BR",
        "ACOES",
        "Dados_Tratados",
        "Dados_Normalizados",
        arq_IBOV,
    )
    df_ibov = pd.read_parquet(ibov)
    cot_ibov = df_ibov.loc[:, "Fech_Ajustado"].to_frame()
    cot_ibov.columns = ["IBOV"]

    # Criando um Benchmark do IBOV e SELIC
    df_benchmark = pd.concat([df_Selic, cot_ibov], axis=1, join="outer")
    df_benchmark = df_benchmark.dropna()
    # (FIM) Criando um Benchmark do IBOV e SELIC

    ## Dados da Expectativa de Selic
    arquivo_Expectativa_Selic = os.path.join(
        "dataset",
        "BR",
        "Selic",
        "Expectativa_Selic_Diaria.parquet",
    )
    df_Expectativa_Selic = pd.read_parquet(arquivo_Expectativa_Selic)

    # Filtrar a data
    condicao_selecao_data = (df_Expectativa_Selic.index <= data_simulacao) & (
        df_Expectativa_Selic.index >= data_inicial
    )
    df_Expectativa_Selic = df_Expectativa_Selic.loc[condicao_selecao_data, :]

    ## (FIM) Dados da Expectativa de Selic

    ## Dados das ações
    dict_df_acoes = {}
    # Exceções
    dict_excecoes = {"PRIO3": datetime(2017, 1, 1)}
    lista_excecoes = list(dict_excecoes.keys())
    lista_excecoes = []
    Lista_nao_encontrados = []
    # DataFrame com as cotações
    df_cotacao_acoes = pd.DataFrame()
    for Ticker in lista_ativos_elegiveis:
        try:
            if Ticker in lista_excecoes:
                data_inicial_acao = dict_excecoes[Ticker]
            else:
                data_inicial_acao = data_inicial

            df_acao, df_multiplos, df_CAGR = Func_ler_dados_acoes(
                Ticker,
                data_inicial_acao,
                data_simulacao,
            )

            df_temp = df_multiplos.sort_index(ascending=True).loc[:, ["Fech_Ajustado"]]
            df_temp.columns = [Ticker]
            df_cotacao_acoes = pd.concat(
                [df_cotacao_acoes, df_temp],
                axis=1,
                join="outer",
            )
            ## Salvando os dados no dicionário
            dict_df_acoes[Ticker] = [df_acao, df_multiplos, df_CAGR]
        except:
            Lista_nao_encontrados.append(Ticker)
            pass

    # Atualizar o df_benchmark
    df_benchmark = pd.concat([df_benchmark, df_cotacao_acoes], axis=1, join="outer")

    return [dict_df_acoes, df_benchmark, df_Expectativa_Selic]
