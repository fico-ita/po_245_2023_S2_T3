"""
Tratamento dos dados que serão utilizados na estratégia
"""
import contextlib
import math
import os

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def endereco_arquivos_brutos(Ticker):
    """Generate paths for raw data files related to a stock ticker.

    The function takes a stock ticker and generates the complete paths for raw data files related to fundamental data, daily quotes,
    dividend data, event data (such as splits and mergers), and subscription data.

    Args:
        Ticker (str): The stock ticker for which the paths of raw files will be generated.

    Returns:
        tuple: A tuple containing the complete paths for raw data files of fundamental data, daily quotes,
        dividend data, event data, and subscription data, respectively.

    Example:
    >>> endereco_arquivos_brutos("EQTL3")
        ('dataset/BR/ACOES/Dados_Brutos/EQTL3_Fund.parquet',
         'dataset/BR/ACOES/Dados_Brutos/EQTL3_Cot.parquet',
         'dataset/BR/ACOES/Dados_Brutos/EQTL3_Prov.parquet',
         'dataset/BR/ACOES/Dados_Brutos/EQTL3_Eventos.parquet',
         'dataset/BR/ACOES/Dados_Brutos/EQTL3_Subscricao.parquet')
    """
    # Dados Fundamentalistas
    ticker_Fund = f"{Ticker}_Fund.parquet"
    arquivo_Fund = os.path.join(
        "..", "..", "dataset", "BR", "ACOES", "Dados_Brutos", ticker_Fund
    )
    # Cotações Diárias
    ticker_Cot = f"{Ticker}_Cot.parquet"
    arquivo_Cot = os.path.join(
        "..", "..", "dataset", "BR", "ACOES", "Dados_Brutos", ticker_Cot
    )
    # Dados de Proventos
    ticker_Prov = f"{Ticker}_Prov.parquet"
    arquivo_Prov = os.path.join(
        "..", "..", "dataset", "BR", "ACOES", "Dados_Brutos", ticker_Prov
    )
    # Dados de Eventos como desdobramentos e grupamentos
    ticker_Eventos = f"{Ticker}_Eventos.parquet"
    arquivo_Eventos = os.path.join(
        "..",
        "..",
        "dataset",
        "BR",
        "ACOES",
        "Dados_Brutos",
        ticker_Eventos,
    )
    # Dados de Subscrições
    ticker_Subscricao = f"{Ticker}_Subscricao.parquet"
    arquivo_Subscricao = os.path.join(
        "..",
        "..",
        "dataset",
        "BR",
        "ACOES",
        "Dados_Brutos",
        ticker_Subscricao,
    )

    return arquivo_Fund, arquivo_Cot, arquivo_Prov, arquivo_Eventos, arquivo_Subscricao


# %%
def endereco_arquivos_normalizados(Ticker):
    """Generate paths for normalized data files related to a stock ticker.

    The function takes a stock ticker and generates the complete paths for normalized data files related to fundamental data and daily quotes.

    Args:
        Ticker (str): The stock ticker for which the paths of normalized files will be generated.

    Returns:
        tuple: A tuple containing the complete paths for normalized data files of fundamental data and daily quotes, respectively.

    Example:
        >>> endereco_arquivos_normalizados("AAPL")
        ('dataset/BR/ACOES/Dados_Tratados/Dados_Normalizados/Dados_normalizados_acao_EQTL3.parquet',
         'dataset/BR/ACOES/Dados_Tratados/Dados_Normalizados/Cotacao_EQTL3.parquet')
    """
    # Dados Fundamentalistas
    ticker_Fund = f"Dados_normalizados_acao_{Ticker}.parquet"
    arquivo_por_acao = os.path.join(
        "..",
        "..",
        "dataset",
        "BR",
        "ACOES",
        "Dados_Tratados",
        "Dados_Normalizados",
        ticker_Fund,
    )
    # Cotações Diárias
    ticker_Cot = f"Cotacao_{Ticker}.parquet"
    arquivo_cotacao = os.path.join(
        "..",
        "..",
        "dataset",
        "BR",
        "ACOES",
        "Dados_Tratados",
        "Dados_Normalizados",
        ticker_Cot,
    )

    return arquivo_por_acao, arquivo_cotacao


def Func_Classe_acao(Ticker):
    """Define the class of a stock based on the ticker code.

    The function takes the ticker of a stock and determines its class based
    on specific characters in the ticker code.

    Args:
        Ticker (str): The stock ticker for which the class will be defined.

    Returns:
        str: The class of the stock, which can be "ON" (Ordinary), "PN" (Preferred),
        "PNA" (Preferred Class A), "PNB" (Preferred Class B), "UNT" (Unit),
        or "ERRO" if the ticker does not match any known class.

    Example:
        >>> Func_Classe_acao("PETR3")
        'ON'
        >>> Func_Classe_acao("ITSA4")
        'PN'
        >>> Func_Classe_acao("VALE5")
        'PNA'
        >>> Func_Classe_acao("BBDC6")
        'PNB'
        >>> Func_Classe_acao("HYPE11")
        'UNT'
        >>> Func_Classe_acao("XYZ123")
        'ERRO'
    """
    if Ticker[4] == "3":
        Classe_acao = "ON"
    elif Ticker[4] == "4":
        Classe_acao = "PN"
    elif Ticker[4] == "5":
        Classe_acao = "PNA"
    elif Ticker[4] == "6":
        Classe_acao = "PNB"
    elif Ticker[4:6] == "11":
        Classe_acao = "UNT"
    else:
        Classe_acao = "ERRO"

    return Classe_acao


def tratar_Fund(arquivo_Fund):
    """Trata os dados fundamentalistas de uma ação.

    A função lê os dados fundamentalistas de um arquivo Parquet, realiza algumas transformações
    nos dados, como tratamento de datas, conversão de strings para float, ajustes de porcentagens
    e correções no preço de fechamento e número de ações. Retorna um DataFrame com os dados tratados.

    Args:
        arquivo_Fund (str): Caminho para o arquivo Parquet contendo os dados fundamentalistas.

    Returns:
        pd.DataFrame: DataFrame com os dados fundamentalistas tratados.

    Example:
        >>> tratar_Fund("dataset/BR/ACOES/Dados_Tratados/Dados_Normalizados/Dados_normalizados_acao_EQTL3.parquet")
        # Retorna um DataFrame com os dados fundamentalistas tratados.
    """
    # Ao ler o arquivo Parquet, especifique o dtype usando o parâmetro dtype
    df_fund = pd.read_parquet(arquivo_Fund)

    # Excluir o que não tem Data_balanco
    df_fund = df_fund.dropna(subset=["Data_balanco", "Num_acoes", "Market_value"])

    # Colunas que são datas
    colunas_datas = [
        "Data_balanco",
        "Data_demonstracao",
        "Data_analise",
    ]  # Substitua pelos nomes reais das suas colunas de datas
    df_fund.loc[:, colunas_datas] = df_fund.loc[:, colunas_datas].apply(
        pd.to_datetime,
        format="%d/%m/%Y",
    )
    df_fund.index = pd.to_datetime(df_fund.index, format="%d/%m/%Y")

    # Suponha que você tenha uma lista de nomes de colunas que devem ser lidas como números
    colunas_numericas = [
        "Num_acoes",
        "Fator_equivalencia_acoes",
        "Fator_cot",
        "Market_value",
        "PL",
        "RL",
        "EBITDA",
        "D&A",
        "EBIT",
        "LL",
        "LL_controlador",
        "LL_nao_controlador",
        "ROIC",
        "ROE",
        "Div_Bruta",
        "Div_liq",
        "Div_Arrendamento",
        "FCO",
        "FCI",
        "FCF",
        "Preco_fechamento",
        "Payout",
        "Proventos",
        "JCP",
        "DY_12m",
        "DY_24m",
        "DY_36m",
        "DY_48m",
        "DY_60m",
        "ret_12meses",
        "ret_1mes_aa",
        "ret_ano",
        "ret_CDI_1m",
        "ret_CDI_12m",
        "ret_CDI_ano",
        "ret_IBOV_1mes",
        "ret_IBOV_12m",
        "ret_IBOV_ano",
        "meses",
    ]  # Substitua pelos nomes reais das suas colunas numéricas
    df_fund = df_fund.fillna(0)
    for col in colunas_numericas:
        # Substitua vírgulas por pontos e converta para float
        with contextlib.suppress(Exception):
            df_fund[col] = df_fund[col].str.replace(",", ".").astype(float)

    df_fund = df_fund.fillna(0)
    df_fund = df_fund[df_fund["Preco_fechamento"] != 0]

    # Colunas que são porcentagem
    colunas_pct = [
        "DY_12m",
        "DY_24m",
        "DY_36m",
        "DY_48m",
        "DY_60m",
        "ret_12meses",
        "ret_1mes_aa",
        "ret_ano",
        "ret_CDI_1m",
        "ret_CDI_12m",
        "ret_CDI_ano",
        "ret_IBOV_1mes",
        "ret_IBOV_12m",
        "ret_IBOV_ano",
    ]  # Substitua pelos nomes reais das suas colunas de porcentagem

    df_fund.loc[:, colunas_pct] = df_fund.loc[:, colunas_pct].apply(lambda x: x / 100)

    ## Ajuste da cotação
    df_fund["Preco_fechamento"] = df_fund["Preco_fechamento"] / df_fund["Fator_cot"]
    ## Ajuste do número de ações
    df_fund["Num_acoes"] = df_fund["Market_value"] / df_fund["Preco_fechamento"]

    df_fund = df_fund.sort_index(ascending=False)
    return df_fund


# %%
def tratar_cot(arquivo_Cot):
    """Trata os dados de cotação de uma ação.

    A função lê os dados de cotação de um arquivo Parquet, realiza algumas transformações
    nos dados, como tratamento de datas, conversão de strings para float, ajustes de porcentagens
    e correções no preço de fechamento. Retorna um DataFrame com os dados tratados.

    Args:
        arquivo_Cot (str): Caminho para o arquivo Parquet contendo os dados de cotação.

    Returns:
        pd.DataFrame: DataFrame com os dados de cotação tratados.

    Example:
        >>> tratar_cot("dataset/BR/ACOES/Dados_Tratados/Dados_Normalizados/Cotacao_EQTL3.parquet")
        # Retorna um DataFrame com os dados de cotação tratados.
    """
    # Lendo os arquivos de cotações
    df_cot = pd.read_parquet(arquivo_Cot)
    # Colunas que são datas
    df_cot.index = pd.to_datetime(df_cot.index, format="%d/%m/%Y")

    # Substitua os valores vazios ("" ou string vazia) por NaN em todas as colunas
    df_cot = df_cot.replace("", np.nan)

    # Excluir o que não tem Fech_Historico
    df_cot = df_cot.dropna(subset=["Fech_Historico"])

    # Colunas que são numéricas
    colunas_numericas = [
        "Fech_Ajustado",
        "Variação(%)",
        "Fech_Historico",
        "Abertura_Ajustado",
        "Min_Ajustado",
        "Medio_Ajustado",
        "Max_Ajustado",
        "Vol(MM_R$)",
        "Negocios",
        "Fator",
    ]
    for col in colunas_numericas:
        # Substitua vírgulas por pontos e converta para float
        try:
            df_cot[col] = df_cot[col].str.replace(",", ".").astype(float)
        except:
            raise

    # Colunas que são porcentagem
    colunas_pct = ["Variação(%)"]
    # Substitua os valores None por NaN em todas as colunas
    df_cot = df_cot.fillna(0)
    df_cot.loc[:, colunas_pct] = df_cot.loc[:, colunas_pct].apply(lambda x: x / 100)

    # Ordenar por index
    df_cot = df_cot.sort_index(ascending=False)

    ## Ajustar a cotação devido ao Fator de Cotação
    df_cot.loc[:, "Fech_Historico"] = df_cot.loc[:, "Fech_Historico"] / df_cot["Fator"]

    return df_cot


# %%
def tratar_prov(arquivo_Prov, Classe_acao):
    """Trata os dados de proventos de uma ação.

    A função lê os dados de proventos de um arquivo Parquet, realiza algumas transformações
    nos dados, como tratamento de datas, conversão de strings para float e filtragem por classe de ação.
    Retorna um DataFrame com os dados tratados e um indicador se existem dados de proventos para a classe
    de ação especificada.

    Args:
        arquivo_Prov (str): Caminho para o arquivo Parquet contendo os dados de proventos.
        Classe_acao (str): Classe da ação para a qual os proventos serão filtrados.

    Returns:
        tuple: Uma tupla contendo um DataFrame com os dados de proventos tratados e um indicador
        (booleano) se existem dados de proventos para a classe de ação especificada.

    Example:
        >>> tratar_prov("dataset/BR/ACOES/Dados_Tratados/Dados_Normalizados/Proventos_EQTL3.parquet", "ON")
        # Retorna uma tupla com um DataFrame com os dados de proventos tratados e um indicador booleano.
    """
    # Lendo os arquivos de proventos
    try:
        Existe_prov = True
        df_prov = pd.read_parquet(arquivo_Prov)

        # Colunas que são datas
        df_prov.index = pd.to_datetime(df_prov.index, format="%d/%m/%Y")

        # Substitua os valores vazios ("" ou string vazia) por NaN em todas as colunas
        df_prov = df_prov.replace("", np.nan)

        # Excluir o que não tem Valor_do_Provento
        df_prov = df_prov.dropna(subset=["Valor_do_Provento"])

        # Colunas que são numéricas
        colunas_numericas = ["Valor_do_Provento", "Último_preco_com", "Provento_por"]
        # Ordenar por index
        df_prov = df_prov.sort_index(ascending=False)
        # Filtra a classe da ação
        df_prov = df_prov.loc[
            (df_prov["Tipo"] == Classe_acao) | (df_prov["Tipo"] == "todas"),
            :,
        ].copy()

        for col in colunas_numericas:
            # Substitua vírgulas por pontos e converta para float
            try:
                df_prov[col] = df_prov[col].str.replace(",", ".").astype(float)
            except:
                raise

    except:
        Existe_prov = False
        df_prov = pd.DataFrame()

    return df_prov, Existe_prov


# %%
def tratar_even(arquivo_Eventos, Classe_acao):
    """Trata os dados de eventos (desdobramentos e grupamentos) de uma ação.

    A função lê os dados de eventos de um arquivo Parquet, realiza algumas transformações
    nos dados, como tratamento de datas, filtragem por classe de ação e conversão de strings para float.
    Retorna um DataFrame com os dados de eventos tratados e um indicador se existem dados de eventos
    para a classe de ação especificada.

    Args:
        arquivo_Eventos (str): Caminho para o arquivo Parquet contendo os dados de eventos.
        Classe_acao (str): Classe da ação para a qual os eventos serão filtrados.

    Returns:
        tuple: Uma tupla contendo um DataFrame com os dados de eventos tratados e um indicador
        (booleano) se existem dados de eventos para a classe de ação especificada.

    Example:
        >>> tratar_even("dataset/BR/ACOES/Dados_Tratados/Dados_Normalizados/Eventos_EQTL3.parquet", "ON")
        # Retorna uma tupla com um DataFrame com os dados de eventos tratados e um indicador booleano.
    """
    # Lendo os arquivos de eventos
    try:
        Existe_eventos = True
        df_eventos = pd.read_parquet(arquivo_Eventos)

        # Colunas que são datas
        df_eventos.index = pd.to_datetime(df_eventos.index, format="%d/%m/%Y")
        colunas_numericas = ["Fator"]
        # Filtra a classe da ação
        df_eventos = df_eventos.loc[
            (df_eventos["ClasseAcao"] == Classe_acao)
            | (df_eventos["ClasseAcao"] == "todas"),
            :,
        ].copy()
        # Ordenar por index
        df_eventos = df_eventos.sort_index(ascending=False)
        # Colunas que são numéricas
        for col in colunas_numericas:
            # Substitua vírgulas por pontos e converta para float
            try:
                df_eventos[col] = df_eventos[col].str.replace(",", ".").astype(float)
            except:
                raise

    except:
        Existe_eventos = False
        df_eventos = pd.DataFrame()

    return df_eventos, Existe_eventos


# %% [markdown]
# ## Normalização


# %%
def normalizar_dados_fund(df_fund, df_eventos, Existe_eventos, Ticker):
    """Normaliza os dados fundamentais considerando o número de ações equivalentes.

    A função recebe um DataFrame contendo dados fundamentais, um DataFrame com dados de eventos
    (desdobramentos, grupamentos) e informações sobre a existência desses eventos. Ela normaliza
    os dados fundamentais pelo número de ações equivalentes, considerando os eventos, e retorna o
    DataFrame tratado.

    Args:
        df_fund (pd.DataFrame): DataFrame contendo dados fundamentais.
        df_eventos (pd.DataFrame): DataFrame contendo dados de eventos.
        Existe_eventos (bool): Indicador se existem eventos para a classe de ação especificada.
        Ticker (str): Ticker da ação para a qual os dados serão normalizados.

    Returns:
        pd.DataFrame: DataFrame contendo os dados fundamentais normalizados pelo número de ações
        equivalentes.

    Example:
        >>> normalizar_dados_fund(df_fund, df_eventos, True, "EQTL3")
        # Retorna um DataFrame com os dados fundamentais normalizados.
    """
    # Normalizar Dados Fundamentalistas e considerar os eventos
    df_Tratar_por_Acao = df_fund.copy()

    ## Calcular o número de ações Equivalentes atual
    # Criar a Coluna de Número de Ações Equivalentes
    df_Tratar_por_Acao.insert(3, "Num_acoes_equivalentes", 0)

    ## No caso de Units é necessário considerar o número de ações de cada classe
    df_Tratar_por_Acao.loc[:, "Num_acoes_equivalentes"] = df_Tratar_por_Acao.loc[
        :,
        "Num_acoes",
    ]

    ## Adicionar o Ticker
    df_Tratar_por_Acao.insert(0, "Ticker", Ticker)

    ## Adicionar o Coluna de Proventos
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_12meses", 0)
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_24meses", 0)
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_36meses", 0)
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_48meses", 0)
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Prov_60meses", 0)

    ## Adicionar os proventos
    df_Tratar_por_Acao.loc[:, "Prov_12meses"] = df_Tratar_por_Acao.apply(
        lambda x: x["DY_12m"] * x["Preco_fechamento"],
        axis=1,
    )
    df_Tratar_por_Acao.loc[:, "Prov_24meses"] = df_Tratar_por_Acao.apply(
        lambda x: x["DY_24m"] * x["Preco_fechamento"],
        axis=1,
    )
    df_Tratar_por_Acao.loc[:, "Prov_36meses"] = df_Tratar_por_Acao.apply(
        lambda x: x["DY_36m"] * x["Preco_fechamento"],
        axis=1,
    )
    df_Tratar_por_Acao.loc[:, "Prov_48meses"] = df_Tratar_por_Acao.apply(
        lambda x: x["DY_48m"] * x["Preco_fechamento"],
        axis=1,
    )
    df_Tratar_por_Acao.loc[:, "Prov_60meses"] = df_Tratar_por_Acao.apply(
        lambda x: x["DY_60m"] * x["Preco_fechamento"],
        axis=1,
    )

    ## Conservar os dados de proventos para considerar apenas os proventos pagos no período
    ## Portanto, 24 meses é 24 meses menos o que foi pago em 12 meses. Assim eu sei o que foi pago no período
    df_Tratar_por_Acao.loc[:, "Prov_60meses"] = (
        df_Tratar_por_Acao.loc[:, "Prov_60meses"]
        - df_Tratar_por_Acao.loc[:, "Prov_48meses"]
    )
    df_Tratar_por_Acao.loc[:, "Prov_48meses"] = (
        df_Tratar_por_Acao.loc[:, "Prov_48meses"]
        - df_Tratar_por_Acao.loc[:, "Prov_36meses"]
    )
    df_Tratar_por_Acao.loc[:, "Prov_36meses"] = (
        df_Tratar_por_Acao.loc[:, "Prov_36meses"]
        - df_Tratar_por_Acao.loc[:, "Prov_24meses"]
    )
    df_Tratar_por_Acao.loc[:, "Prov_24meses"] = (
        df_Tratar_por_Acao.loc[:, "Prov_24meses"]
        - df_Tratar_por_Acao.loc[:, "Prov_12meses"]
    )

    # Computar os desdobramentos, grupamentos e Bonificações
    if Existe_eventos:
        ## Datas do Eventos com a condição de Classe de Ação
        datas_eventos = df_eventos.loc[:].index.unique()

        ## Loop para percorrer as datas dos eventos e alterar o número equivalente de ações
        for data in datas_eventos:
            ## Fator do evento
            condicao_even = df_eventos.index == data
            fator_evento = df_eventos.loc[condicao_even, "Fator"].prod()

            ## Condição para recalcular o número de ações equivalentes, no df_Tratar_por_Acao
            ## Lembrar que "data" é a "data-com" do evento, portanto está incluída
            condicao = df_Tratar_por_Acao.index < data
            ## Recalcular o número de ações equivalentes
            df_Tratar_por_Acao.loc[condicao, "Num_acoes_equivalentes"] = (
                df_Tratar_por_Acao.loc[condicao, "Num_acoes_equivalentes"]
                / fator_evento
            )

            ## Recalcular os proventos
            df_Tratar_por_Acao.loc[condicao, "Prov_12meses"] = (
                df_Tratar_por_Acao.loc[condicao, "Prov_12meses"] * fator_evento
            )
            df_Tratar_por_Acao.loc[condicao, "Prov_24meses"] = (
                df_Tratar_por_Acao.loc[condicao, "Prov_24meses"] * fator_evento
            )
            df_Tratar_por_Acao.loc[condicao, "Prov_36meses"] = (
                df_Tratar_por_Acao.loc[condicao, "Prov_36meses"] * fator_evento
            )
            df_Tratar_por_Acao.loc[condicao, "Prov_48meses"] = (
                df_Tratar_por_Acao.loc[condicao, "Prov_48meses"] * fator_evento
            )
            df_Tratar_por_Acao.loc[condicao, "Prov_60meses"] = (
                df_Tratar_por_Acao.loc[condicao, "Prov_60meses"] * fator_evento
            )

    ## Adicionar o Ticker
    df_Tratar_por_Acao.insert(len(df_Tratar_por_Acao.columns), "Fonte", "Comdinheiro")

    ## Normalizar os dados pelo número de ações equivalentes
    colunas_divididas = [
        "PL",
        "RL",
        "EBITDA",
        "D&A",
        "EBIT",
        "LL",
        "LL_controlador",
        "LL_nao_controlador",
        "Div_Bruta",
        "Div_liq",
        "Div_Arrendamento",
        "FCO",
        "FCI",
        "FCF",
    ]
    for col in colunas_divididas:
        df_Tratar_por_Acao.loc[:, col] = df_Tratar_por_Acao.apply(
            lambda row: row[col] / row["Num_acoes_equivalentes"],
            axis=1,
        )

    return df_Tratar_por_Acao


# %%
def ajuste_cotacoes(df_cot, df_Tratar_por_Acao, df_eventos, Existe_eventos, Ticker):
    """Trata os dados de cotações diante de eventos, ajustando o número de ações equivalentes.

    A função recebe um DataFrame com dados de cotações, um DataFrame contendo dados fundamentais
    tratados, um DataFrame com dados de eventos (desdobramentos, grupamentos), informações sobre a
    existência desses eventos e o Ticker da ação. A função ajusta os dados de cotações considerando
    os eventos e retorna um DataFrame tratado.

    Args:
        df_cot (pd.DataFrame): DataFrame contendo dados de cotações.
        df_Tratar_por_Acao (pd.DataFrame): DataFrame contendo dados fundamentais tratados.
        df_eventos (pd.DataFrame): DataFrame contendo dados de eventos.
        Existe_eventos (bool): Indicador se existem eventos para a classe de ação especificada.
        Ticker (str): Ticker da ação para a qual os dados serão ajustados.

    Returns:
        pd.DataFrame: DataFrame contendo os dados de cotações ajustados.

    Example:
        >>> ajuste_cotacoes(df_cot, df_Tratar_por_Acao, df_eventos, True, "EQTL3")
        # Retorna um DataFrame com os dados de cotações ajustados.
    """
    ## Tratar o dataframe das cotações com os eventos
    df_cot_tratado = df_cot.loc[:, ["Fech_Historico", "Fech_Ajustado"]].copy()
    df_cot_tratado.columns = ["Fechamento_Equivalente", "Fech_Ajustado"]
    df_cot_tratado.insert(0, "Num_acoes_equivalentes", 0)

    ## Filtrar as datas do df_Tratar_por_Acao menores que a data do df_cot_tratado
    datas_fund = df_Tratar_por_Acao.index
    datas_cot = df_cot_tratado.index
    ## Loop para percorrer as datas dos eventos e alterar o número equivalente de ações
    ## Considerando a Classe da ação. Lembrar que o Num_acoes_equivalentes calculado anteriormente
    ## só foi calculado para cada trimestre, aqui considera o número de ações equivalentes para cada dia
    for data in datas_cot:
        condicao = datas_fund < data
        # Se todos forem falso pode deletar a linha
        if condicao.sum() > 0:
            Num_acoes_equi = df_Tratar_por_Acao.loc[condicao, "Num_acoes_equivalentes"][
                0
            ]
            # Fator_equivalencia_acoes = df_Tratar_por_Acao.loc[condicao,"Fator_equivalencia_acoes"][0]
            df_cot_tratado.loc[data, "Num_acoes_equivalentes"] = Num_acoes_equi
            # print(data, Num_acoes, Fator_equivalencia_acoes)
        else:
            df_cot_tratado = df_cot_tratado.drop(data)

    ## Normalizar os dados pelo número de ações equivalentes, considerando os eventos

    if Existe_eventos:
        datas_eventos = df_eventos.index.unique()
        for data in datas_eventos:
            ## Fator do evento
            condicao_evento = df_eventos.index == data
            fator_evento = df_eventos.loc[condicao_evento, "Fator"].prod()

            ## Condição para recalcular o Preço de Fechamento Histórico, no df_cot_tratado
            condicao = df_cot_tratado.index <= data
            ## Recalcular o preço Equivalente
            df_cot_tratado.loc[condicao, "Fechamento_Equivalente"] = (
                df_cot_tratado.loc[condicao, "Fechamento_Equivalente"] * fator_evento
            )

            ## Recalcular o número de ações equivalentes
            # df_cot_tratado.loc[condicao,"Num_acoes_equivalentes"] = df_cot_tratado.loc[condicao,"Num_acoes_equivalentes"]/fator_evento

    ## Adicionar o Ticker
    df_cot_tratado.insert(0, "Ticker", Ticker)
    # Adicionar o Coluna de Market Value
    df_cot_tratado.insert(len(df_cot_tratado.columns), "Market_value", 0)
    df_cot_tratado.loc[:, "Market_value"] = df_cot_tratado.apply(
        lambda row: row["Fechamento_Equivalente"] * row["Num_acoes_equivalentes"],
        axis=1,
    )

    # Fonte
    df_cot_tratado.insert(len(df_cot_tratado.columns), "Fonte", "Comdinheiro")

    return df_cot_tratado


# %% [markdown]
# ## Anualizar Dados Fundamentalistas


# %%
def Func_Anualizar_Dados_Func(df_Tratar_por_Acao):
    """Trata os dados de cotações diante de eventos, ajustando o número de ações equivalentes.

    A função recebe um DataFrame com dados de cotações, um DataFrame contendo dados fundamentais
    tratados, um DataFrame com dados de eventos (desdobramentos, grupamentos), informações sobre a
    existência desses eventos e o Ticker da ação. A função ajusta os dados de cotações considerando
    os eventos e retorna um DataFrame tratado.

    Args:
        df_cot (pd.DataFrame): DataFrame contendo dados de cotações.
        df_Tratar_por_Acao (pd.DataFrame): DataFrame contendo dados fundamentais tratados.
        df_eventos (pd.DataFrame): DataFrame contendo dados de eventos.
        Existe_eventos (bool): Indicador se existem eventos para a classe de ação especificada.
        Ticker (str): Ticker da ação para a qual os dados serão ajustados.

    Returns:
        pd.DataFrame: DataFrame contendo os dados de cotações ajustados.

    Example:
        >>> ajuste_cotacoes(df_cot, df_Tratar_por_Acao, df_eventos, True, "EQTL3")
        # Retorna um DataFrame com os dados de cotações ajustados.
    """
    ## Anualizar os dados fundamentalistas
    # Supondo que seu DataFrame esteja ordenado por datas, caso contrário, ordene-o primeiro
    df_Tratar_por_Acao = df_Tratar_por_Acao.sort_index()

    # Defina o tamanho da janela para 4 trimestres (um ano)
    window_size = 4
    df_anual = df_Tratar_por_Acao.copy()

    # Crie um novo DataFrame para armazenar os valores anuais
    df_anual.loc[:, "RL":"LL_nao_controlador"] = (
        df_Tratar_por_Acao.loc[:, "RL":"LL_nao_controlador"]
        .rolling(window=window_size)
        .sum()
    )
    df_anual.loc[:, "FCO":"FCF"] = (
        df_Tratar_por_Acao.loc[:, "FCO":"FCF"].rolling(window=window_size).sum()
    )

    ## Adicionar o EV
    df_anual.loc[:, "EV"] = (
        df_Tratar_por_Acao.loc[:, "Preco_fechamento"]
        + df_Tratar_por_Acao.loc[:, "Div_liq"]
    )

    ## Consertando o ROE
    df_anual.loc[:, "ROE"] = df_anual.loc[:, "LL"] / df_anual.loc[:, "PL"]

    # Adionar a margem Líquida
    df_anual.loc[:, "ML"] = df_anual.loc[:, "LL"] / df_anual.loc[:, "RL"]

    # Drop NaNs
    df_anual = df_anual.dropna()

    df_anual = df_anual.sort_index(ascending=False)

    return df_anual


# %% [markdown]
# ## Múltiplos

# %% [markdown]
# ### Funções Auxilares

# %%
# Defina uma função que calcula o número de indicações de alta entre todas as combinações


def calcular_indicacoes_alta(row):
    """Calcula o percentual de indicações de alta entre todas as combinações de médias móveis.

    Args:
        row (pd.Series): Uma linha do DataFrame contendo as médias móveis.

    Returns:
        float: O percentual de indicações de alta.

    Example:
        >>> calcular_indicacoes_alta(df.iloc[0])
        0.75
    """
    indicacoes = 0
    colunas = row.loc["MM1":"MM36"].index
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
def Func_weighted_mean_alta(row, taxa_exp):
    """Calcula a média ponderada dos valores em uma linha de dados com base nas datas.

    Args:
        row (pd.Series): Uma linha do DataFrame contendo os valores.
        taxa_exp (float): Taxa de decaimento exponencial.

    Returns:
        float: A média ponderada.

    Example:
        >>> Func_weighted_mean_alta(df.iloc[0], 0.85)
        0.75
    """
    # Última Data
    data_simulacao = row.index.max()
    datas = row.index

    # Calcule as diferenças em dias entre cada data da simulação e a cada data
    time_diff = (data_simulacao - datas).days
    time_diff = np.array(time_diff)

    # Calcule os pesos com base na redução em 0.85 a cada 365 dias
    weights_serie = taxa_exp ** (time_diff / 365)

    # Calcule a média ponderada e o desvio padrão ponderado
    weighted_mean = np.average(row, weights=weights_serie)

    return weighted_mean


# %% [markdown]
# #### load data backtest


# %%
def load_data_backtest(df_multiplos_diarios_anual):
    """Carrega dados de multiplicadores diários anuais e calcula uma variedade de indicadores financeiros.

    Parâmetros:
    - df_multiplos_diarios_anual (pd.DataFrame): DataFrame contendo dados financeiros diários anuais.

    Retorna:
    - pd.DataFrame: DataFrame com dados originais e indicadores financeiros calculados.
    """
    cotacoes = df_multiplos_diarios_anual.loc[:, ["Fech_Ajustado"]]
    cotacoes = cotacoes.sort_index(ascending=True)
    # Calcular rentabilidade dos últimos 12 meses
    cotacoes["rent_72_b"] = (
        cotacoes["Fech_Ajustado"] / cotacoes["Fech_Ajustado"].shift(252 * 6)
    ) ** (1 / 6) - 1
    cotacoes["rent_36_b"] = (
        cotacoes["Fech_Ajustado"] / cotacoes["Fech_Ajustado"].shift(252 * 3)
    ) ** (1 / 3) - 1
    cotacoes["rent_12_b"] = (
        cotacoes["Fech_Ajustado"] / cotacoes["Fech_Ajustado"].shift(252) - 1
    )
    cotacoes["rent_6_b"] = (
        cotacoes["Fech_Ajustado"] / cotacoes["Fech_Ajustado"].shift(21 * 6)
    ) ** 2 - 1
    cotacoes["rent_3_b"] = (
        cotacoes["Fech_Ajustado"] / cotacoes["Fech_Ajustado"].shift(21 * 3)
    ) ** 4 - 1
    cotacoes["rent_1_b"] = (
        cotacoes["Fech_Ajustado"] / cotacoes["Fech_Ajustado"].shift(21)
    ) ** 12 - 1

    # Calcular rentabilidade dos próximos x meses (x meses à frente)
    cotacoes["rent_1_f"] = (
        cotacoes["Fech_Ajustado"].shift(-21) / cotacoes["Fech_Ajustado"]
    ) ** 12 - 1
    cotacoes["rent_3_f"] = (
        cotacoes["Fech_Ajustado"].shift(-21 * 3) / cotacoes["Fech_Ajustado"]
    ) ** 4 - 1
    cotacoes["rent_6_f"] = (
        cotacoes["Fech_Ajustado"].shift(-21 * 6) / cotacoes["Fech_Ajustado"]
    ) ** 2 - 1
    cotacoes["rent_12_f"] = (
        cotacoes["Fech_Ajustado"].shift(-252) / cotacoes["Fech_Ajustado"] - 1
    )
    cotacoes["rent_36_f"] = (
        cotacoes["Fech_Ajustado"].shift(-252 * 3) / cotacoes["Fech_Ajustado"]
    ) ** (1 / 3) - 1
    cotacoes["rent_72_f"] = (
        cotacoes["Fech_Ajustado"].shift(-252 * 6) / cotacoes["Fech_Ajustado"]
    ) ** (1 / 6) - 1

    # Cálculo de Média Móveis
    cotacoes["MM1"] = cotacoes["Fech_Ajustado"].rolling(window=21, min_periods=1).mean()

    cotacoes["MM3"] = (
        cotacoes["Fech_Ajustado"].rolling(window=21 * 3, min_periods=1).mean()
    )

    cotacoes["MM6"] = (
        cotacoes["Fech_Ajustado"].rolling(window=21 * 6, min_periods=1).mean()
    )

    cotacoes["MM9"] = (
        cotacoes["Fech_Ajustado"].rolling(window=21 * 9, min_periods=1).mean()
    )

    cotacoes["MM12"] = (
        cotacoes["Fech_Ajustado"].rolling(window=21 * 12, min_periods=1).mean()
    )

    cotacoes["MM18"] = (
        cotacoes["Fech_Ajustado"].rolling(window=21 * 18, min_periods=1).mean()
    )

    cotacoes["MM24"] = (
        cotacoes["Fech_Ajustado"].rolling(window=21 * 24, min_periods=1).mean()
    )

    cotacoes["MM36"] = (
        cotacoes["Fech_Ajustado"].rolling(window=21 * 36, min_periods=1).mean()
    )

    # Aplique a função a cada linha do DataFrame e crie uma nova coluna
    cotacoes["Indicacoes_Alta"] = cotacoes.apply(calcular_indicacoes_alta, axis=1)

    # Calcular a média dessas Indicações de Alta
    serie_alta = cotacoes["Indicacoes_Alta"]
    serie_alta = serie_alta.sort_index(ascending=True)
    cotacoes["Indicacoes_Alta_media"] = serie_alta.expanding(min_periods=2).apply(
        Func_weighted_mean_alta,
        args=(0.8,),
    )

    df_multiplos_diarios_anual_2 = pd.concat(
        [
            df_multiplos_diarios_anual,
            cotacoes.loc[:, "rent_72_b":"Indicacoes_Alta_media"],
        ],
        axis=1,
    )
    df_multiplos_diarios_anual_2 = df_multiplos_diarios_anual_2.sort_index(
        ascending=False,
    )

    return df_multiplos_diarios_anual_2


# %% [markdown]
# #### Média e desvio padrão dos Múltiplos


# %%
def truncate_series(values, std_dev=3, years_limit=5):
    """Calcula a média ponderada e o desvio padrão ponderado de uma série temporal, removendo outliers.

    Parâmetros:
    - values (pd.Series): Série temporal para a qual a média e o desvio padrão serão calculados.
    - std_dev (float, opcional): Número de desvios padrão considerados para identificar outliers. Padrão é 3.
    - years_limit (int, opcional): Número de anos a serem considerados para calcular a média e o desvio padrão. Padrão é 5.

    Retorna:
    - tuple: Média ponderada e desvio padrão ponderado da série sem outliers.
    """
    # Última Data
    data_simulacao = values.index.max()
    # Index dos últimos years_limit anos
    datas_5years = values.index > data_simulacao - pd.DateOffset(years=years_limit)
    serie_Teste_5years = values.loc[datas_5years]
    # Calcular a média e desvio padrão dos últimos years_limit anos
    media = serie_Teste_5years.mean()
    std = serie_Teste_5years.std()

    # Verificar os outliers
    outliers = values[np.abs(values - media) > std_dev * std]
    outliers_distant = outliers.index[
        outliers.index < values.index.max() - pd.DateOffset(years=years_limit)
    ]

    # Remova os valores identificados
    serie_cleaned = values.drop(outliers_distant)

    datas = serie_cleaned.index
    data_simulacao = serie_cleaned.index.max()

    # Calcule as diferenças em dias entre cada data da simulação e a cada data
    time_diff = (data_simulacao - datas).days
    time_diff = np.array(time_diff)
    # Calcule os pesos com base na redução em 0.85 a cada 365 dias
    weights_serie = 0.85 ** (time_diff / 365)

    # Calcule a média ponderada e o desvio padrão ponderado

    weighted_mean = np.average(serie_cleaned, weights=weights_serie)
    weighted_var = np.average(
        (serie_cleaned - weighted_mean) ** 2,
        weights=weights_serie,
    )
    weighted_std = np.sqrt(weighted_var)

    return weighted_mean, weighted_std


def Func_weighted_mean_and_std(row, dado, std_dev, years_limit):
    """Calcula a média ponderada ou o desvio padrão ponderado de uma linha em um DataFrame, removendo outliers.

    Parâmetros:
    - row (pd.Series): Linha do DataFrame contendo os dados para os quais a média e o desvio padrão serão calculados.
    - dado (str): Tipo de dado a ser calculado ('media' para média ponderada, 'std' para desvio padrão ponderado).
    - std_dev (float): Número de desvios padrão considerados para identificar outliers.
    - years_limit (int): Número de anos a serem considerados para calcular a média e o desvio padrão.

    Retorna:
    - float: Média ponderada ou desvio padrão ponderado da linha sem outliers.

    Raises:
    - Exception: Lança uma exceção se houver um erro ao calcular a média ponderada e o desvio padrão ponderado.
    """
    try:
        weighted_mean, weighted_std = truncate_series(row, std_dev, years_limit)
    except:
        # print("Erro ao calcular a média ponderada e o desvio padrão ponderado.")
        raise

    if dado == "media":
        result = weighted_mean
    elif dado == "std":
        result = weighted_std

    return result


# %%
def Func_mean_std_multiplos(df_multiplos_diarios):
    """Calcula a média, o desvio padrão e o desvio em relação à média de múltiplos financeiros.

    Parâmetros:
    - df_multiplos_diarios (pd.DataFrame): DataFrame contendo dados de múltiplos financeiros.

    Retorna:
    - pd.DataFrame: DataFrame atualizado com colunas de médias, desvios padrão e desvios em relação à média.

    Descrição:
    A função calcula as médias, desvios padrão e desvios em relação à média para múltiplos financeiros específicos.
    Os múltiplos incluídos são 'EV_SR', 'EV_EBITDA', 'PE', 'Margem_liquida' e 'Margem_EBITDA'. A média e o desvio
    padrão são calculados de forma ponderada, removendo outliers, e a coluna de desvio é criada em relação à média.
    Além disso, é calculada uma dívida líquida EBITDA normalizada diante da média de margem EBITDA.

    Raises:
    - Exception: Pode levantar uma exceção se houver algum erro durante o cálculo.

    Exemplo:
    df_multiplos_diarios = Func_mean_std_multiplos(df_multiplos_diarios)
    """
    df_multiplos_diarios_2 = df_multiplos_diarios.copy()

    lista_multiplos = ["EV_SR", "EV_EBITDA", "PE", "Margem_liquida", "Margem_EBITDA"]
    for multiplo in lista_multiplos:
        serie_multiplo = df_multiplos_diarios_2.loc[:, multiplo].copy()
        serie_multiplo = serie_multiplo.sort_index(ascending=True)
        serie_multiplo = serie_multiplo.dropna()
        datas_dif_zero = serie_multiplo.ne(0)
        primeira_ocorrencia_nao_nula = serie_multiplo.loc[datas_dif_zero].index.min()

        serie_multiplo = serie_multiplo.loc[
            serie_multiplo.index >= primeira_ocorrencia_nao_nula
        ]

        media_mult = serie_multiplo.expanding(min_periods=252).apply(
            Func_weighted_mean_and_std,
            args=("media", 4, 4),
        )
        std_mult = serie_multiplo.expanding(min_periods=252).apply(
            Func_weighted_mean_and_std,
            args=("std", 4, 4),
        )

        desvio_mult = (serie_multiplo - media_mult) / std_mult

        coluna_media = f"{multiplo}_media"
        coluna_std = f"{multiplo}_std"
        coluna_desvio = f"{multiplo}_desvio"

        # Salvar os dados
        df_multiplos_diarios_2[coluna_media] = media_mult
        df_multiplos_diarios_2[coluna_std] = std_mult
        df_multiplos_diarios_2[coluna_desvio] = desvio_mult

    # Calculando uma dívida líquida EBITDA normalizado diante da média
    # de margem EBITDA
    df_temp = df_multiplos_diarios_2.loc[
        :,
        ["DIV_liq_EBITDA", "Margem_EBITDA", "Margem_EBITDA_media"],
    ]

    df_multiplos_diarios_2["DIV_liq_EBITDA_medio"] = df_temp.apply(
        lambda row: row["DIV_liq_EBITDA"]
        if (row["DIV_liq_EBITDA"] >= 10) or (row["Margem_EBITDA_media"] <= 0)
        else row["DIV_liq_EBITDA"]
        * (
            (row["Margem_EBITDA"])
            / (row["Margem_EBITDA"] * 0.3 + row["Margem_EBITDA_media"] * 0.7)
        ),
        axis=1,
    )

    return df_multiplos_diarios_2


# %% [markdown]
# #### Cálculo dos múltiplos hitóricos


# %%
def multiplos_diarios(df_Tratar_por_Acao, df_cot_tratado, Ticker):
    """Calcula os múltiplos financeiros diários com base nos dados de balanço e cotação.

    Parâmetros:
    - df_Tratar_por_Acao (pd.DataFrame): DataFrame contendo dados fundamentalistas.
    - df_cot_tratado (pd.DataFrame): DataFrame contendo dados de cotações.
    - Ticker (str): Símbolo do ticker da ação.

    Retorna:
    - pd.DataFrame: DataFrame com múltiplos financeiros calculados diariamente.

    Descrição:
    A função calcula múltiplos financeiros diários com base nos dados de balanço e cotação.
    Os múltiplos incluídos são EV, EV arrendado, PVPA, EV/SR, PSR, EV/EBITDA, EV/EBITDA arrendado,
    P/EBIT, P/E, P/E controlador, FCO, FCI, FCF, ROE, Margem líquida, Margem EBITDA, Dívida Bruta/PL,
    Dívida Líquida/EBITDA, Dívida Líquida Arrendamento/EBITDA, DY 12 meses, DY 24 meses, DY 36 meses,
    DY 48 meses, DY 60 meses e DY médio.

    Raises:
    - Exception: Pode levantar uma exceção se houver algum erro durante o cálculo.

    Exemplo:
    df_multiplos_diarios = multiplos_diarios(df_Tratar_por_Acao, df_cot_tratado, 'EQTL3')
    """
    # Coletando as datas de balanço e cotação
    data_cotacao = df_cot_tratado.index
    data_balanco = df_Tratar_por_Acao.index[0:-4]
    if len(data_balanco) < 2:
        return pd.DataFrame()
    primeiro_balanco = data_balanco[-1]
    data_multiplos_diarios = data_cotacao[data_cotacao >= primeiro_balanco]

    # Criando um DataFrame de múltiplos com as datas de balanço e cotação, Trimetre e Anual
    df_multiplos_diarios = pd.DataFrame(
        index=data_multiplos_diarios,
        columns=[
            "Num_acoes_equivalentes",
            "Fechamento_Equivalente",
            "Fech_Ajustado",
            "Market_value",
            "EV",
            "EV_arrend",
            "PVPA",
            "EV_SR",
            "PSR",
            "EV_EBITDA",
            "EV_EBITDA_Arr",
            "P_EBIT",
            "PE",
            "PE_C",
            "FCO",
            "FCI",
            "FCF",
            "ROE",
            "Margem_liquida",
            "Margem_EBITDA",
            "DIV_Bruta_PL",
            "DIV_liq_EBITDA",
            "DIV_Arrendamento_EBITDA",
            "DY_12m",
            "DY_24m",
            "DY_36m",
            "DY_48m",
            "DY_60m",
            "DY_medio",
            "Fonte",
        ],
    )

    ## Preenchendo os dados de cotação
    df_multiplos_diarios.insert(0, "Ticker", Ticker)
    df_multiplos_diarios.loc[:, "Fonte"] = "Comdinheiro"

    # Preenchendo alguns valores
    lista_colunas = [
        "Num_acoes_equivalentes",
        "Fechamento_Equivalente",
        "Fech_Ajustado",
        "Market_value",
    ]
    df_multiplos_diarios.loc[:, lista_colunas] = df_cot_tratado.loc[:, lista_colunas]

    # Preenchendo os valores Absolutos
    for data in data_multiplos_diarios:
        ## Não pegar dados futuros! Pois 'Data_balanco' é a data de divulgação do balanço
        condicao = df_Tratar_por_Acao.loc[:, "Data_balanco"] <= data
        # print(data, condicao)
        if condicao.sum() > 0:
            # Data Balanco
            data_Balanco = df_Tratar_por_Acao.loc[condicao, :].index.max()

            ## Preço Equivalente
            Preco_equivalente = df_multiplos_diarios.loc[data, "Fechamento_Equivalente"]
            ## EV = Market_value + Div_liq; Atenção que não foi considerado a dívida de arrendamento
            EV = Preco_equivalente + df_Tratar_por_Acao.loc[data_Balanco, "Div_liq"]
            EV_Arrend = EV + df_Tratar_por_Acao.loc[data_Balanco, "Div_Arrendamento"]

            # Para estabilidade do EV
            if Preco_equivalente / 2 >= EV:
                EV = Preco_equivalente / 2

            ## Atribuindo os valores de EV
            df_multiplos_diarios.loc[data, "EV"] = EV
            df_multiplos_diarios.loc[data, "EV_arrend"] = EV_Arrend

            ## Dados Fundamentalistas
            PL = df_Tratar_por_Acao.loc[data_Balanco, "PL"]
            Receita_12meses = df_Tratar_por_Acao.loc[data_Balanco, "RL"]
            EBITDA_12meses = df_Tratar_por_Acao.loc[data_Balanco, "EBITDA"]
            EBIT_12meses = df_Tratar_por_Acao.loc[data_Balanco, "EBIT"]
            LL_12meses = df_Tratar_por_Acao.loc[data_Balanco, "LL"]
            LL_C_12meses = df_Tratar_por_Acao.loc[data_Balanco, "LL_controlador"]
            Div_liq_12meses = df_Tratar_por_Acao.loc[data_Balanco, "Div_liq"]
            Div_bruta_12meses = df_Tratar_por_Acao.loc[data_Balanco, "Div_Bruta"]
            Div_liq_Arr_12meses = (
                Div_liq_12meses
                + df_Tratar_por_Acao.loc[data_Balanco, "Div_Arrendamento"]
            )
            FCO_12meses = df_Tratar_por_Acao.loc[data_Balanco, "FCO"]
            FCI_12meses = df_Tratar_por_Acao.loc[data_Balanco, "FCI"]
            FCF_12meses = df_Tratar_por_Acao.loc[data_Balanco, "FCF"]

            ## Registrando os dados de 1 ano
            df_multiplos_diarios.loc[data, "PVPA"] = (
                Preco_equivalente / PL if PL != 0 else 0
            )
            df_multiplos_diarios.loc[data, "EV_SR"] = (
                EV / Receita_12meses if Receita_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "PSR"] = (
                Preco_equivalente / Receita_12meses if Receita_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "EV_EBITDA"] = (
                EV / EBITDA_12meses if EBITDA_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "EV_EBITDA_Arr"] = (
                EV_Arrend / EBITDA_12meses if EBITDA_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "P_EBIT"] = (
                Preco_equivalente / EBIT_12meses if EBIT_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "PE"] = (
                Preco_equivalente / LL_12meses if LL_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "PE_C"] = (
                Preco_equivalente / LL_C_12meses if LL_C_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "FCO"] = (
                Preco_equivalente / FCO_12meses if FCO_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "FCI"] = (
                Preco_equivalente / FCI_12meses if FCI_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "FCF"] = (
                Preco_equivalente / FCF_12meses if FCF_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "ROE"] = LL_12meses / PL if PL > 0 else -1
            df_multiplos_diarios.loc[data, "Margem_liquida"] = (
                LL_12meses / Receita_12meses if Receita_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "Margem_EBITDA"] = (
                EBITDA_12meses / Receita_12meses if Receita_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "DIV_Bruta_PL"] = (
                Div_bruta_12meses / PL if PL != 0 else 0
            )
            df_multiplos_diarios.loc[data, "DIV_liq_EBITDA"] = (
                Div_liq_12meses / EBITDA_12meses if EBITDA_12meses != 0 else 0
            )
            df_multiplos_diarios.loc[data, "DIV_liq_EBITDA"] = (
                Div_liq_12meses / EBITDA_12meses if EBITDA_12meses > 0 else 10
            )
            df_multiplos_diarios.loc[data, "DIV_Arrendamento_EBITDA"] = (
                Div_liq_Arr_12meses / EBITDA_12meses if EBITDA_12meses != 0 else 0
            )

            ## Adicionar os proventos
            # Proventos
            Prov_1ano = df_Tratar_por_Acao.loc[data_Balanco, "Prov_12meses"]
            Prov_2ano = df_Tratar_por_Acao.loc[data_Balanco, "Prov_24meses"]
            Prov_3ano = df_Tratar_por_Acao.loc[data_Balanco, "Prov_36meses"]
            Prov_4ano = df_Tratar_por_Acao.loc[data_Balanco, "Prov_48meses"]
            Prov_5ano = df_Tratar_por_Acao.loc[data_Balanco, "Prov_60meses"]

            ## Salvando o DY médio

            df_multiplos_diarios.loc[data, "DY_12m"] = Prov_1ano / Preco_equivalente
            df_multiplos_diarios.loc[data, "DY_24m"] = Prov_2ano / (Preco_equivalente)
            df_multiplos_diarios.loc[data, "DY_36m"] = Prov_3ano / (Preco_equivalente)
            df_multiplos_diarios.loc[data, "DY_48m"] = Prov_4ano / (Preco_equivalente)
            df_multiplos_diarios.loc[data, "DY_60m"] = Prov_5ano / (Preco_equivalente)

            df_multiplos_diarios.loc[data, "DY_medio"] = (
                Prov_1ano + Prov_2ano + Prov_3ano + Prov_4ano + Prov_5ano
            ) / (Preco_equivalente * 5)

    ## Dados para o back-test
    df_multiplos_diarios = load_data_backtest(df_multiplos_diarios)

    # Referência dos Múltiplos
    df_multiplos_diarios = Func_mean_std_multiplos(df_multiplos_diarios)
    return df_multiplos_diarios


# %% [markdown]
# ## Funções para estimar o crescimento

# %% [markdown]
# ### Funções Auxiliares para CAGR


# %%
def CalculoCAGRnAnos(df_cresc, Fundamento):
    """Calcula o CAGR (Compound Annual Growth Rate) de um dado fundamentalista ao longo de diferentes períodos.

    Parâmetros:
    - df_cresc (DataFrame): DataFrame contendo dados fundamentais e de tempo.
    - Fundamento (str): Nome do dado fundamentalista a ser analisado.

    Retorna:
    - Tuple: Uma tupla contendo o CAGR para períodos de 8, 6, 4, 2 anos, o menor CAGR entre eles,
             a previsão do valor do fundamento e a previsão mínima.

    Descrição:
    A função calcula o CAGR (Compound Annual Growth Rate) de um dado fundamentalista ao longo de diferentes períodos,
    incluindo 8, 6, 4 e 2 anos. Também retorna o menor CAGR entre esses períodos, a previsão do valor do fundamento
    e a previsão mínima. O cálculo é baseado em ajustes de curvas polinomiais e considera o logaritmo do valor do fundamento
    para lidar com números negativos.

    Raises:
    - Exception: Pode levantar uma exceção se houver algum erro durante o cálculo.

    Exemplo:
    CAGR_8, CAGR_6, CAGR_4, CAGR_2, CAGR_min, Previsao, Previsao_minima = CalculoCAGRnAnos(df_fundamental, 'Receita')
    """
    # Última data
    tempo_ultimo = df_cresc.loc[df_cresc.index.max(), "Tempo"]

    # Cópia do df
    df_cresc_8 = df_cresc.copy()

    # Selecionando o Fundamento que será utilizado
    df_cresc_8 = df_cresc_8.loc[:, ["Tempo", Fundamento, "Tipo_Balanco"]]

    # Selecionando o tipo_Balanco
    # df_cresc_8 = df_cresc_8[df_cresc_8['Tipo_Balanco']=="consolidado"]

    # Primeiro índice positivo
    indices_fund_pos = df_cresc_8[df_cresc_8[Fundamento] > 0].index
    menor_indice_pos = indices_fund_pos.min()

    # Selecionando o df
    df_cresc_8 = df_cresc_8[df_cresc_8.index > menor_indice_pos]

    # Selecionado outros intervalos de ano 2 ano
    df_cresc_completo = df_cresc_8[df_cresc_8["Tempo"] >= tempo_ultimo - 2]

    # Pelo menos os últimos dois anos completos
    if len(df_cresc_completo) < 9:
        return 0, 0, 0, 0, 0, 0, 0

    # Pegando alguns valores importantes
    menor_valor = df_cresc_8[Fundamento].min()
    maior_valor = df_cresc_8[Fundamento].max()
    media_valor = df_cresc_8[Fundamento].mean()
    ultm_valor = df_cresc_8.loc[df_cresc_8.index.max(), Fundamento]

    # Solução para lidar com números negativos
    if menor_valor < 0:
        df_cresc_8[Fundamento] = (
            df_cresc_8[Fundamento]
            - menor_valor
            + min(max(media_valor / 2, maior_valor / 20), maior_valor / 5)
        )

    # Aplicando Ln
    df_cresc_8["Ln"] = df_cresc_8[Fundamento].apply(lambda x: np.log(x))

    # Selecionado outros intervalos tempo, 6, 4, 2 anos

    df_cresc_2 = df_cresc_8[df_cresc_8["Tempo"] >= tempo_ultimo - 2]
    df_cresc_4 = df_cresc_8[df_cresc_8["Tempo"] >= tempo_ultimo - 4]
    df_cresc_6 = df_cresc_8[df_cresc_8["Tempo"] >= tempo_ultimo - 6]
    try:
        # Highest power first
        expoente_8 = (
            np.polyfit(df_cresc_8["Tempo"].values, df_cresc_8["Ln"].values, 1)[0]
            * (len(df_cresc_8) / (8 * 4 + 1)) ** 2
        )
        expoente_6 = (
            np.polyfit(df_cresc_6["Tempo"].values, df_cresc_6["Ln"].values, 1)[0]
            * (len(df_cresc_6) / (6 * 4 + 1)) ** 2
        )
        expoente_4 = (
            np.polyfit(df_cresc_4["Tempo"].values, df_cresc_4["Ln"].values, 1)[0]
            * (len(df_cresc_4) / (4 * 4 + 1)) ** 2
        )
        expoente_2 = (
            np.polyfit(df_cresc_2["Tempo"].values, df_cresc_2["Ln"].values, 1)[0]
            * (len(df_cresc_2) / (2 * 4 + 1)) ** 2
        )
    except:
        expoente_8 = 0
        expoente_6 = 0
        expoente_4 = 0
        expoente_2 = 0

    ## Crescimento composto anualizado
    Cagr_8 = (np.exp(expoente_8)) - 1
    Cagr_6 = (np.exp(expoente_6)) - 1
    Cagr_4 = (np.exp(expoente_4)) - 1
    Cagr_2 = (np.exp(expoente_2)) - 1
    Cagr_min = min(Cagr_8, Cagr_6, Cagr_4, Cagr_2)

    # Previsão do valor da fundamento
    Previsao = ultm_valor * (1 + Cagr_8)
    Previsao_minima = ultm_valor * (1 + Cagr_min)

    return Cagr_8, Cagr_6, Cagr_4, Cagr_2, Cagr_min, Previsao, Previsao_minima


# %%
def Func_inicar_df_CAGR(df_Tratar_por_Acao, Fund):
    df_CAGR = pd.DataFrame(
        index=df_Tratar_por_Acao.index,
        columns=[
            "Ticker",
            "Fund",
            "Fund_Valor",
            "CAGR8",
            "CAGR6",
            "CAGR4",
            "CAGR2",
            "CAGR_min",
            "Fund_Prev",
            "Fund_Prev_min",
            "Data_balanco",
            "Ordenado",
            "Fonte",
        ],
    )

    Ticker = df_Tratar_por_Acao["Ticker"][0]
    df_CAGR.loc[:, ["Ticker", "Fund", "Ordenado", "Fonte"]] = (
        Ticker,
        Fund,
        0,
        "Comdinheiro",
    )
    df_CAGR.loc[:, "Fund_Valor"] = df_Tratar_por_Acao.loc[:, Fund]
    df_CAGR.loc[:, "Data_balanco"] = pd.to_datetime(
        df_Tratar_por_Acao.loc[:, "Data_balanco"],
    ).dt.date

    return df_CAGR


def Func_atraso_balanco(row):
    data_balanco = row["Data_balanco"]

    data_max_balanco = row["data"] + pd.DateOffset(months=3)
    data_2_balancos = row["data"] + pd.DateOffset(months=6)

    if (data_balanco > data_max_balanco) and (data_balanco < data_2_balancos):
        new_row = row.copy()
        new_row["Data_balanco"] = data_max_balanco
        new_row["CAGR_min"] = -1
        return new_row  # Retorna a linha modificada

    # Definir 'Data_balanco' como 'None' quando não há atraso
    row["Data_balanco"] = None
    return row


def Func_apply_tempo_discreto(row):
    data = row.name
    ano = data.year
    mes = data.month
    tempo = ano + mes / 12
    return tempo


def Func_ver_ordenado(row):
    serie = row["CAGR8":"CAGR2"]

    if serie.equals(serie.sort_values()):
        ordenado = 1
    elif serie.equals(serie.sort_values(ascending=False)):
        ordenado = -1
    else:
        ordenado = 0
    return ordenado


# %%
def Func_Criar_CAGR_Medio(df_CAGR, df_acao):
    # Lista dos Fundamentos que serão realizados uma média
    lista = ["RL", "LL", "EBITDA"]
    df_temp = df_CAGR.loc[df_CAGR["Fund"].isin(lista)]
    # Criando outro df
    colunas = df_temp.columns
    df_CAGR_medio = pd.DataFrame(columns=colunas)

    # Agrupando o DataFrame pelo campo 'data'
    grouped = df_temp.groupby("data")
    for data, grupo in grouped:
        # Dados Iniciais
        ticker = grupo["Ticker"].iloc[0]
        fund = "Medio"
        fund_Valor = 0
        fund_Prev = 0
        Fund_Prev_min = 0
        data_balanco = grupo["Data_balanco"].iloc[0]
        fonte = grupo["Fonte"].iloc[0]
        # SubsubSetor
        SubSubsetor = df_acao["SubSubsetor"][0]

        # Selecionando as linhas
        row_RL = grupo[grupo["Fund"] == "RL"]
        row_EBITDA = grupo[grupo["Fund"] == "EBITDA"]
        row_LL = grupo[grupo["Fund"] == "LL"]

        if SubSubsetor == "Bancos":
            serie_CAGR = (
                row_RL.loc[:, "CAGR8":"CAGR2"] * 0.6
                + row_LL.loc[:, "CAGR8":"CAGR2"] * 0.4
            )
        else:
            serie_CAGR = (
                row_RL.loc[:, "CAGR8":"CAGR2"] * 0.4
                + row_EBITDA.loc[:, "CAGR8":"CAGR2"] * 0.3
                + row_LL.loc[:, "CAGR8":"CAGR2"] * 0.3
            )

        df_CAGR_medio.loc[data, "Ticker"] = ticker
        df_CAGR_medio.loc[data, "Fund"] = fund
        df_CAGR_medio.loc[data, "Fund_Valor"] = fund_Valor
        df_CAGR_medio.loc[data, "Fund_Prev"] = fund_Prev
        df_CAGR_medio.loc[data, "Fund_Prev_min"] = Fund_Prev_min
        df_CAGR_medio.loc[data, "Data_balanco"] = data_balanco
        df_CAGR_medio.loc[data, "Fonte"] = fonte
        df_CAGR_medio.loc[data, "CAGR8":"CAGR2"] = serie_CAGR.loc[data]
        df_CAGR_medio.loc[data, "CAGR_min"] = serie_CAGR.loc[data].values.min()

    # Set index as date of the first 'balance sheet'
    df_CAGR_medio = df_CAGR_medio.reset_index()
    df_CAGR_medio = df_CAGR_medio.rename(columns={"index": "data"})

    return df_CAGR_medio


# %% [markdown]
# ### Função Principal


# %%
def Func_CAGR(df_Tratar_por_Acao, Ticker):
    # Cópia do df
    df_Tratar_por_Acao_new = df_Tratar_por_Acao.copy()

    # Discretizar o tempo
    df_Tratar_por_Acao_new["Tempo"] = df_Tratar_por_Acao_new.apply(
        Func_apply_tempo_discreto,
        axis=1,
    )

    # Initialize the DataFrames
    df_CAGR_PL = Func_inicar_df_CAGR(df_Tratar_por_Acao, "PL")
    df_CAGR_RL = Func_inicar_df_CAGR(df_Tratar_por_Acao, "RL")
    df_CAGR_EBITDA = Func_inicar_df_CAGR(df_Tratar_por_Acao, "EBITDA")
    df_CAGR_LL = Func_inicar_df_CAGR(df_Tratar_por_Acao, "LL")
    df_CAGR_Proventos = Func_inicar_df_CAGR(df_Tratar_por_Acao, "Prov_12meses")

    for data in df_Tratar_por_Acao_new.index:
        # Pegar o último valores de tempo
        tempo_ultimo = df_Tratar_por_Acao_new.loc[data, "Tempo"]
        tempo_inicio = tempo_ultimo - 8
        condicao = (df_Tratar_por_Acao_new["Tempo"] >= tempo_inicio) & (
            df_Tratar_por_Acao_new["Tempo"] <= tempo_ultimo
        )
        df_Crescimento = df_Tratar_por_Acao_new.loc[
            condicao,
            ["Tempo", "RL", "EBITDA", "LL", "Prov_12meses", "Tipo_Balanco"],
        ]

        ## Cálculo do CAGR Receita Líquida
        if len(df_Crescimento) > 4 * 5:
            ## Cálculo do CAGR Receita Líquida
            df_CAGR_RL.loc[
                data,
                [
                    "CAGR8",
                    "CAGR6",
                    "CAGR4",
                    "CAGR2",
                    "CAGR_min",
                    "Fund_Prev",
                    "Fund_Prev_min",
                ],
            ] = CalculoCAGRnAnos(df_Crescimento, "RL")

            ## Cálculo do CAGR EBITDA
            df_CAGR_EBITDA.loc[
                data,
                [
                    "CAGR8",
                    "CAGR6",
                    "CAGR4",
                    "CAGR2",
                    "CAGR_min",
                    "Fund_Prev",
                    "Fund_Prev_min",
                ],
            ] = CalculoCAGRnAnos(df_Crescimento, "EBITDA")

            ## Cálculo do CAGR Lucro Líquido
            df_CAGR_LL.loc[
                data,
                [
                    "CAGR8",
                    "CAGR6",
                    "CAGR4",
                    "CAGR2",
                    "CAGR_min",
                    "Fund_Prev",
                    "Fund_Prev_min",
                ],
            ] = CalculoCAGRnAnos(df_Crescimento, "LL")

            ## Cálculo do CAGR Proventos
            df_CAGR_Proventos.loc[
                data,
                [
                    "CAGR8",
                    "CAGR6",
                    "CAGR4",
                    "CAGR2",
                    "CAGR_min",
                    "Fund_Prev",
                    "Fund_Prev_min",
                ],
            ] = CalculoCAGRnAnos(df_Crescimento, "Prov_12meses")

            # Ajuste para o dividendos mínimo
            # Pegar a média dos últimos 3 anos
            Prov_36meses = df_Tratar_por_Acao_new.loc[
                data,
                "Prov_12meses":"Prov_36meses",
            ].mean()

            # Pegar o mínimo calculado
            Fund_Prev_min = df_CAGR_Proventos.loc[data, "Fund_Prev_min"]

            # Calcular o novo mínimo
            df_CAGR_Proventos.loc[data, "Fund_Prev_min"] = min(
                Prov_36meses,
                Fund_Prev_min,
            )

        else:
            ## Cálculo do CAGR Receita Líquida
            df_CAGR_RL.loc[
                data,
                [
                    "CAGR8",
                    "CAGR6",
                    "CAGR4",
                    "CAGR2",
                    "CAGR_min",
                    "Fund_Prev",
                    "Fund_Prev_min",
                ],
            ] = 0

            ## Cálculo do CAGR EBITDA
            df_CAGR_EBITDA.loc[
                data,
                [
                    "CAGR8",
                    "CAGR6",
                    "CAGR4",
                    "CAGR2",
                    "CAGR_min",
                    "Fund_Prev",
                    "Fund_Prev_min",
                ],
            ] = 0

            ## Cálculo do CAGR Lucro Líquido
            df_CAGR_LL.loc[
                data,
                [
                    "CAGR8",
                    "CAGR6",
                    "CAGR4",
                    "CAGR2",
                    "CAGR_min",
                    "Fund_Prev",
                    "Fund_Prev_min",
                ],
            ] = 0

            ## Cálculo do CAGR Proventos
            df_CAGR_Proventos.loc[
                data,
                [
                    "CAGR8",
                    "CAGR6",
                    "CAGR4",
                    "CAGR2",
                    "CAGR_min",
                    "Fund_Prev",
                    "Fund_Prev_min",
                ],
            ] = 0

    # Cálculo do CAGR para PL
    df_CAGR_PL.loc[:, "Fund_Prev"] = (
        df_CAGR_PL.loc[:, "Fund_Valor"] + df_CAGR_LL.loc[:, "Fund_Prev"]
    )
    df_CAGR_PL.loc[:, "CAGR8"] = (
        df_CAGR_PL.loc[:, "Fund_Prev"] / df_CAGR_PL.loc[:, "Fund_Valor"] - 1
    )

    # Return the index to ascending order False
    df_CAGR_RL = df_CAGR_RL.sort_index(ascending=False)

    # Fill na with zero
    df_CAGR_RL = df_CAGR_RL.fillna(0)

    # Concatenate the dataframes
    df_CAGR = pd.concat(
        [df_CAGR_PL, df_CAGR_RL, df_CAGR_EBITDA, df_CAGR_LL, df_CAGR_Proventos],
        axis=0,
    )

    # Calculando um CAGR do Fund - Médio
    df_CAGR.loc[:, "Data_balanco"] = pd.to_datetime(df_CAGR.loc[:, "Data_balanco"])

    # Calcular o df_CAGR_medio
    df_CAGR_medio = Func_Criar_CAGR_Medio(df_CAGR, df_Tratar_por_Acao_new)

    # Set index as date of the first 'balance sheet'
    df_CAGR = df_CAGR.reset_index()

    new_rows = df_CAGR.loc[df_CAGR["Fund"] == "RL"].apply(Func_atraso_balanco, axis=1)

    # Removendo as linhas onde 'Data_balanco' é 'None'
    new_rows = new_rows.dropna(subset=["Data_balanco"])

    # Adicionando as novas linhas
    df_CAGR = pd.concat([df_CAGR, new_rows, df_CAGR_medio])

    # Ordenando o DataFrame pela coluna 'data' e 'Data_balanco'
    df_CAGR = df_CAGR.sort_values(by=["data", "Data_balanco"], ascending=False)
    df_CAGR = df_CAGR.set_index("Data_balanco")

    # Fill na with zero
    df_CAGR = df_CAGR.fillna(0)
    df_CAGR.index = pd.to_datetime(df_CAGR.index)

    # Verificar se os valore de crescimento estão ordenados
    df_CAGR.loc[
        df_CAGR.index.year >= df_CAGR.index.min().year + 6,
        "Ordenado",
    ] = df_CAGR.loc[df_CAGR.index.year >= df_CAGR.index.min().year + 6, :].apply(
        Func_ver_ordenado,
        axis=1,
    )

    return df_CAGR


# %% [markdown]
# ## Função para tratar dados


# %%
def Tratar_dados_diarios_Normalizados(Ticker):
    ## Endereço dos arquivos
    (
        arquivo_Fund,
        arquivo_Cot,
        arquivo_Prov,
        arquivo_Eventos,
        arquivo_Subscricao,
    ) = endereco_arquivos_brutos(Ticker)

    print(f"Entrou em {Ticker}")
    ## Verificando a classe da ação
    Classe_acao = Func_Classe_acao(Ticker)

    ## Tratar dados Fundamentalistas
    df_fund = tratar_Fund(arquivo_Fund)

    ## Tratar dados de cotaçõesGR6	CAGR4	CAGR2	CAGR_min	Fund_Prev	Fund_Prev_min	Fonte	index
    df_cot = tratar_cot(arquivo_Cot)

    ## Tratar dados de proventos
    # df_prov, Existe_prov = tratar_prov(arquivo_Prov, Classe_acao)

    ## Tratar dados de eventos
    df_eventos, Existe_eventos = tratar_even(arquivo_Eventos, Classe_acao)

    ## Normalizar Dados Fundamentalistas e considerar os eventos
    df_Tratar_por_Acao = normalizar_dados_fund(
        df_fund,
        df_eventos,
        Existe_eventos,
        Ticker,
    )

    ## Ajustar as cotações
    df_cot_tratado = ajuste_cotacoes(
        df_cot,
        df_Tratar_por_Acao,
        df_eventos,
        Existe_eventos,
        Ticker,
    )

    ## Anualizar os dados
    df_Tratar_por_Acao_anual = Func_Anualizar_Dados_Func(df_Tratar_por_Acao)

    ### Salvar os arquivos em parquet
    # Salvar Fundamentos
    nome_arquivo = f"Dados_normalizados_acao_{Ticker}.parquet"
    arquivo_por_acao = os.path.join(
        "..",
        "..",
        "dataset",
        "BR",
        "ACOES",
        "Dados_Tratados",
        "Dados_Normalizados",
        nome_arquivo,
    )
    # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
    df_Tratar_por_Acao_anual = df_Tratar_por_Acao_anual.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    df_Tratar_por_Acao_anual.to_parquet(arquivo_por_acao, engine="fastparquet")

    ## Salvar Cotação diários
    nome_arquivo = f"Cotacao_{Ticker}.parquet"
    arquivo_cot = os.path.join(
        "..",
        "..",
        "dataset",
        "BR",
        "ACOES",
        "Dados_Tratados",
        "Dados_Normalizados",
        nome_arquivo,
    )
    # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
    df_cot_tratado.to_parquet(arquivo_cot, engine="fastparquet")

    return df_Tratar_por_Acao_anual, df_cot_tratado


# %%
def Tratar_dados_diarios_Projetado(Ticker, atu_mult=True, atu_CAGR=False):
    ## Endereço dos arquivos
    arquivo_por_acao, arquivo_cotacao = endereco_arquivos_normalizados(Ticker)

    df_Tratar_por_Acao = pd.read_parquet(arquivo_por_acao)
    df_cot_tratado = pd.read_parquet(arquivo_cotacao)

    if atu_mult is True:
        ## Multiplos diários
        df_multiplos_diarios_anual = multiplos_diarios(
            df_Tratar_por_Acao,
            df_cot_tratado,
            Ticker,
        )
        ### Salvar os arquivos em parquet
        ## Salvar Múltiplos diários
        nome_arquivo = f"Multiplos_diarios_{Ticker}.parquet"
        arquivo_multiplos = os.path.join(
            "..",
            "..",
            "dataset",
            "BR",
            "ACOES",
            "Dados_Tratados",
            "Dados_Projetados",
            nome_arquivo,
        )
        # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
        df_multiplos_diarios_anual = df_multiplos_diarios_anual.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        df_multiplos_diarios_anual.to_parquet(arquivo_multiplos, engine="fastparquet")
    else:
        nome_arquivo = f"Multiplos_diarios_{Ticker}.parquet"
        arquivo_multiplos = os.path.join(
            "..",
            "..",
            "dataset",
            "BR",
            "ACOES",
            "Dados_Tratados",
            "Dados_Projetados",
            nome_arquivo,
        )
        df_multiplos_diarios_anual = pd.read_parquet(arquivo_multiplos)

    if atu_CAGR is True:
        ## CAGR
        df_CAGR = Func_CAGR(df_Tratar_por_Acao, Ticker)
        ### Salvar os arquivos em parquet
        ## Salvar o CAGR
        nome_arquivo = f"CAGR_{Ticker}.parquet"
        arquivo_CAGR = os.path.join(
            "..",
            "..",
            "dataset",
            "BR",
            "ACOES",
            "Dados_Tratados",
            "Dados_Projetados",
            nome_arquivo,
        )
        # Substitua os valores infinitos por NaN (ou qualquer outro valor desejado)
        df_CAGR = df_CAGR.replace([np.inf, -np.inf], np.nan)
        df_CAGR.to_parquet(arquivo_CAGR, engine="fastparquet")

    else:
        nome_arquivo = f"CAGR_{Ticker}.parquet"
        arquivo_CAGR = os.path.join(
            "..",
            "..",
            "dataset",
            "BR",
            "ACOES",
            "Dados_Tratados",
            "Dados_Projetados",
            nome_arquivo,
        )
        df_CAGR = pd.read_parquet(arquivo_CAGR)

    return df_multiplos_diarios_anual, df_CAGR


# %%
def Tratar_dados_diarios(Ticker, atu_normalizados, atu_mult=True, atu_CAGR=False):
    if atu_normalizados is True:
        df_Tratar_por_Acao, df_cot_tratado = Tratar_dados_diarios_Normalizados(Ticker)

    else:
        # Salvar Fundamentos
        nome_arquivo = f"Dados_normalizados_acao_{Ticker}.parquet"
        arquivo_por_acao = os.path.join(
            "..",
            "..",
            "dataset",
            "BR",
            "ACOES",
            "Dados_Tratados",
            "Dados_Normalizados",
            nome_arquivo,
        )
        df_Tratar_por_Acao = pd.read_parquet(arquivo_por_acao)
        pd.DataFrame()

    df_multiplos_diarios_anual, df_CAGR = Tratar_dados_diarios_Projetado(
        Ticker,
        atu_mult,
        atu_CAGR,
    )
    return df_Tratar_por_Acao, df_multiplos_diarios_anual, df_CAGR


# %% [markdown]
# ### Lendo os arquivos dos dados Brutos

# %%


# %% [markdown]
# # Realizar o tratamento dos dados

# %% [markdown]
# ## Tratamento das ações


def Tratar_dados_fund(lista_ativos):
    ## Ler os ativos que serão buscados
    arquivo_busca = os.path.join(
        "..",
        "..",
        "dataset",
        "BR",
        "ACOES",
        "Dados_Brutos",
        "Lista_Ativos_Busca.parquet",
    )
    Lista_ativos_busca = pd.read_parquet(arquivo_busca)
    Lista_ativos_busca = Lista_ativos_busca["Ticker"].to_list()

    arquivo_busca = os.path.join("..", "..", "dataset", "BR", "ACOES", "IBrA.parquet")
    Lista_ativos_busca2 = pd.read_parquet(arquivo_busca)
    Lista_ativos_busca2 = Lista_ativos_busca2.index.to_list()

    Ativos_adicionais = [
        "GGBR3",
        "GOAU3",
        "ITSA3",
        "KLBN3",
        "SAPR3",
        "TAEE3",
        "USIM3",
        "LVTC3",
        "MTRE3",
    ]

    ## Juntar as listas
    Lista_ativos_busca_Total = (
        Lista_ativos_busca + Lista_ativos_busca2 + Ativos_adicionais
    )
    # Converta a lista em um conjunto (set) para obter valores únicos
    Lista_ativos_busca_Total = set(Lista_ativos_busca_Total)

    # O conjunto (set) agora contém valores únicos
    # Se desejar, converta o conjunto de volta em uma lista
    Lista_ativos_busca_Total = list(Lista_ativos_busca_Total)

    # A lista agora contém valores únicos
    Lista_ativos_busca = Lista_ativos_busca_Total
    Lista_ativos_busca.sort()
    Lista_ativos_nao_deu_certo = []

    if lista_ativos == ["Todos"]:
        lista_iteracao = Lista_ativos_busca
    else:
        lista_iteracao = lista_ativos

    for Ticker in lista_iteracao:
        try:
            (
                df_Tratar_por_Acao,
                df_multiplos_diarios_anual,
                df_CAGR,
            ) = Tratar_dados_diarios(Ticker, True, True, True)
            print(f"Deu certo {Ticker}")
        except:
            print(f"Não deu certo {Ticker}")
            Lista_ativos_nao_deu_certo.append(Ticker)
            pass
