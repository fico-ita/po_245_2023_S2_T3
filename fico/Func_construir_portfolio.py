"""
Constrói uma alocação de um portfólio com ativos
do Ibovespa para um determinado dia
"""
import contextlib
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from pypfopt import (
    DiscreteAllocation,
    EfficientFrontier,
    objective_functions,
    risk_models,
)


def load_data_main(data_simulacao, IBOV=True):
    """Load financial data for simulation.

    This function uses the `load_data` module to load financial data for simulation.
    It returns a list containing a dictionary of stock data, a benchmark dataframe, and a monthly
    Selic expectation dataframe.

    Args:
        data_simulacao (str): Simulation date in the format 'YYYY-MM-DD'.
        IBOV (bool, optional): Flag to include IBOV data. Default is True.

    Returns:
        list: A list containing three elements - a dictionary of stock data, a benchmark dataframe,
        and a monthly Selic expectation dataframe.

    Example:
        load_data_main("2023-01-01", True)
    """
    # Importar a Biblioteca
    from fico.load_data import load_data

    # Carregando os dados
    [dict_df_acoes_m, df_benchmark_m, df_Expectativa_Selic_mensal_m] = load_data(
        data_simulacao, IBOV
    )

    return [dict_df_acoes_m, df_benchmark_m, df_Expectativa_Selic_mensal_m]


# %% [markdown]
# ### Fundamentos objetivo setorial

# %%
global setores_sensivel_juros
setores_sensivel_juros = [
    "Alimentos",
    "Eletrodomésticos",
    "Exploração de Imóveis",
    "Exploração de Rodovias",
    "Incorporações",
    "Linhas Aéreas de Passageiros",
    "Madeira",
    "Máq. e Equip. Industriais",
    "Material Aeronáutico e de Defesa",
    "Material Rodoviário",
    "Produtos de Cuidado Pessoal",
    "Produtos Diversos",
    "Serviços Educacionais",
    "Serviços Financeiros Diversos",
    "Tecidos, Vestuário e Calçados",
    "Viagens e Turismo",
]


def Func_Plotar_Portfolio(df_alocacao):
    """Generate a pie chart to visualize the composition of a portfolio based on an allocation DataFrame.

    Parameters:
        df_alocacao (DataFrame): A DataFrame containing asset allocation in a portfolio. It should have columns named 'leftover' and 'Total'.

    Returns:
        None


    """
    # Copia o DataFrame de alocação para evitar modificações indesejadas
    df_portifolio_dia = df_alocacao.copy()

    # Seleciona a primeira linha do DataFrame para obter a alocação do portfólio no dia
    serie_port = df_portifolio_dia.iloc[0, :]

    # Remove as colunas 'leftover' e 'Total', pois elas não fazem parte da composição do portfólio
    serie_port = serie_port.drop(["leftover", "Total"])

    # Gera um gráfico de pizza para visualizar a composição da carteira
    serie_port.plot(
        figsize=(15, 6),
        kind="pie",
        title="Composição da Carteira",
        autopct="%.2f%%",
    )

    # Não há valor de retorno, pois o gráfico é exibido diretamente na saída


def Func_ajuste_reta(x_fit, x_inf, x_sup, y_inf, y_sup):
    """Perform linear interpolation between two points and calculate the corresponding y value for x_fit.

    Parameters:
        x_fit (float): The value of x for which to calculate y_fit.
        x_inf (float): The x value of the lower point.
        x_sup (float): The x value of the upper point.
        y_inf (float): The y value corresponding to the lower point.
        y_sup (float): The y value corresponding to the upper point.

    Returns:
        y_fit (float): The adjusted y value for x_fit.

    This function performs linear interpolation between two points (x_inf, y_inf) and (x_sup, y_sup) and calculates the corresponding y value for a given x value (x_fit) within that interval. If x_fit is less than x_inf, the returned value will be y_inf. If x_fit is greater than x_sup, the returned value will be y_sup. For x_fit values within the interval (x_inf, x_sup), the function calculates the y value using the equation of the line fitted between the lower and upper points.

    Example:

        Func_ajuste_reta(1, 2, 4, 3, 7)

        3.0
    """
    # Dados de exemplo
    x = np.array([x_inf, x_sup])
    y = np.array([y_inf, y_sup])

    # Ajuste um polinômio de segundo grau (parábola)
    coefficients = np.polyfit(x, y, 1)

    # Crie uma função polinomial com base nos coeficientes
    polynomial = np.poly1d(coefficients)

    ## Limites
    if x_fit <= x_inf:
        y_fit = y_inf
    elif x_fit >= x_sup:
        y_fit = y_sup
    else:
        # Calcule o valor correspondentes de y para o ajuste polinomial
        y_fit = polynomial(x_fit)
        y_fit = round(y_fit, 3)

    return y_fit


def Func_fator_juros(ind_aumento_selic, limite_inferior, limite_superior, Setor):
    """Calculate the penalty factor based on the increase in the Selic rate for companies in different sectors.

    Parameters:
        ind_aumento_selic (float): Indicator of the increase in the Selic rate.
        limite_inferior (float): Lower limit for the Selic rate increase indicator.
        limite_superior (float): Upper limit for the Selic rate increase indicator.
        Setor (str): Sector of the company.

    Returns:
        float: Calculated penalty factor.

    Description:
    The function calculates the penalty factor based on the increase in the Selic rate for companies in different sectors.
    The calculation considers different linear adjustments based on the defined limits and the Selic rate increase indicator.
    The penalty factor is returned.

    Example:
        fator_penalizacao = Func_fator_juros(0.3, 0.1, 0.7, 'Bancos')
    """
    if Setor in setores_sensivel_juros:
        if ind_aumento_selic <= 0.4:
            penal_j = Func_ajuste_reta(ind_aumento_selic, limite_inferior, 0.4, 1.1, 1)
        else:
            penal_j = Func_ajuste_reta(ind_aumento_selic, 0.4, limite_superior, 1, 0.2)
    else:
        penal_j = Func_ajuste_reta(
            ind_aumento_selic,
            limite_inferior,
            limite_superior,
            1.1,
            0.9,
        )

    return penal_j


def Dados_iniciais(df_acao, df_multiplos, df_CAGR, df_benchmark):
    """Collects initial data necessary for the calculation of other functions.

    Args:
        df_acao (pd.DataFrame): DataFrame containing fundamental data of the stock normalized by the number of ex-treasury shares.
        df_multiplos (pd.DataFrame): DataFrame containing company multiples (daily data).
        df_CAGR (pd.DataFrame): DataFrame containing the compound annual growth rate (CAGR) of fundamental data.
        df_benchmark (pd.DataFrame): DataFrame containing benchmark data.

    Returns:
        pd.Series: Series of collected initial data for use in other functions.

    The function collects information such as sector, ROE, average DY, ticker, average CAGR, current multiple, average multiple,
    multiple deviation, and multiple expansion. These data are used in other functions for analysis.

    Example:
        initial_data = Dados_iniciais(df_acao, df_multiplos, df_CAGR, df_benchmark)
    """
    # Última Selic
    ultm_SELIC = df_benchmark.loc[:, "anula100"] / 100
    ultm_SELIC = ultm_SELIC.dropna()
    ultm_SELIC = ultm_SELIC.loc[ultm_SELIC.index.max()]

    # Select CAGR data
    df_CAGR_Medio = df_CAGR.loc[df_CAGR["Fund"] == "Medio", :]
    df_CAGR_Prov = df_CAGR.loc[df_CAGR["Fund"] == "Prov_12meses", :]
    df_CAGR_RL = df_CAGR.loc[df_CAGR["Fund"] == "RL", :]
    df_CAGR_EBITDA = df_CAGR.loc[df_CAGR["Fund"] == "EBITDA", :]
    df_CAGR_LL = df_CAGR.loc[df_CAGR["Fund"] == "LL", :]

    ## Data da simulação alterada
    data_simulacao = df_multiplos.index.max()
    data_ultm_balanco = df_acao.index.max()
    data_ultm_CAGR_Medio = df_CAGR_Medio.index.max()
    data_ultm_CAGR_Prov = df_CAGR_Prov.index.max()
    data_ultm_CAGR_RL = df_CAGR_RL.index.max()
    data_ultm_CAGR_EBITDA = df_CAGR_EBITDA.index.max()
    data_ultm_CAGR_LL = df_CAGR_LL.index.max()

    ## Preço médio x meses
    preco_atual_36 = df_multiplos.loc[
        df_multiplos.index >= data_simulacao - pd.DateOffset(months=36),
        "Fechamento_Equivalente",
    ].mean()

    ## Retorno médio
    rent = df_multiplos.loc[data_simulacao, "rent_72_b":"rent_72_f"]

    ## Coletando dados iniciais
    Setor = df_acao.loc[data_ultm_balanco, "SubSubsetor"]
    ticker = df_acao["Ticker"][0]

    ### Margem Líquida
    ML_ultm = df_multiplos.loc[data_simulacao, "Margem_liquida"]
    ML_media = df_multiplos.loc[data_simulacao, "Margem_liquida_media"]
    ML_std = df_multiplos.loc[data_simulacao, "Margem_liquida_std"]
    ML_1std = ML_media - ML_std

    ### Margem EBITDA
    M_EBITDA_media = df_multiplos.loc[data_simulacao, "Margem_EBITDA_media"]
    M_EBITDA_std = df_multiplos.loc[data_simulacao, "Margem_EBITDA_std"]
    M_EBITDA_1std = M_EBITDA_media - M_EBITDA_std

    ### Dívida Líquida EBITDA
    Dl_EBITDA = df_multiplos.loc[data_simulacao, "DIV_liq_EBITDA_medio"]

    ## Múltiplos
    ### Receita
    EV_SR_desvio = df_multiplos.loc[data_simulacao, "EV_SR_desvio"]

    ### EBITDA
    EV_EBITDA_desvio = df_multiplos.loc[data_simulacao, "EV_EBITDA_desvio"]

    ## CAGR
    CAGR_Medio_8 = df_CAGR_Medio.loc[data_ultm_CAGR_Medio, "CAGR8"]
    CAGR_Medio_6 = df_CAGR_Medio.loc[data_ultm_CAGR_Medio, "CAGR6"]
    CAGR_Medio_4 = df_CAGR_Medio.loc[data_ultm_CAGR_Medio, "CAGR4"]
    CAGR_Medio_2 = df_CAGR_Medio.loc[data_ultm_CAGR_Medio, "CAGR2"]
    CAGR_ordenado = df_CAGR_Medio.loc[data_ultm_CAGR_Medio, "Ordenado"]
    CAGR_Medio_min = df_CAGR_Medio.loc[data_ultm_CAGR_Medio, "CAGR_min"]

    CAGR_MIN_Receita = df_CAGR_RL.loc[data_ultm_CAGR_RL, "CAGR_min"]
    if CAGR_MIN_Receita == -1:  ## Balanço em atraso
        CAGR_Medio_8, CAGR_Medio_6, CAGR_Medio_4, CAGR_Medio_2 = 0, 0, 0, 0
    else:
        CAGR_Receita_2 = df_CAGR_RL.loc[data_ultm_CAGR_RL, "CAGR2"]
        CAGR_Ebitda_2 = df_CAGR_EBITDA.loc[data_ultm_CAGR_EBITDA, "CAGR2"]
        CAGR_Ll_2 = df_CAGR_LL.loc[data_ultm_CAGR_LL, "CAGR2"]
        if Setor == "Bancos":
            CAGR_MIN_Receita = CAGR_Receita_2 * 0.7 + CAGR_Ll_2 * 0.3
        else:
            CAGR_MIN_Receita = CAGR_Receita_2 * 0.7 + CAGR_Ebitda_2 * 0.3

    ## Dividendos
    Prov_prev = df_CAGR_Prov.loc[data_ultm_CAGR_Prov, "Fund_Prev"]
    DY_prev = Prov_prev / preco_atual_36
    Prov_prev_min = df_CAGR_Prov.loc[data_ultm_CAGR_Prov, "Fund_Prev_min"]
    DY_min = Prov_prev_min / preco_atual_36

    ## Fator Técnico, MM
    Ind_alta_MM = df_multiplos.loc[data_simulacao, "Indicacoes_Alta"]
    Ind_alta_MM_media = df_multiplos.loc[data_simulacao, "Indicacoes_Alta_media"]

    ## Compactando os dados iniciais em um dicionário
    lista_dados = [
        ticker,
        Setor,
        ML_ultm,
        ML_1std,
        M_EBITDA_1std,
        Dl_EBITDA,
        EV_SR_desvio,
        EV_EBITDA_desvio,
        CAGR_Medio_8,
        CAGR_Medio_6,
        CAGR_Medio_4,
        CAGR_Medio_2,
        CAGR_ordenado,
        CAGR_Medio_min,
        CAGR_MIN_Receita,
        DY_prev,
        DY_min,
        Ind_alta_MM,
        Ind_alta_MM_media,
        ultm_SELIC,
    ]
    lista_indices = [
        "ticker",
        "Setor",
        "ML_ultm",
        "ML_1std",
        "M_EBITDA_1std",
        "Dl_EBITDA",
        "EV_SR_desvio",
        "EV_EBITDA_desvio",
        "CAGR_Medio_8",
        "CAGR_Medio_6",
        "CAGR_Medio_4",
        "CAGR_Medio_2",
        "CAGR_ordenado",
        "CAGR_Medio_min",
        "CAGR_MIN_Receita",
        "DY_prev",
        "DY_min",
        "Ind_alta_MM",
        "Ind_alta_MM_media",
        "ultm_SELIC",
    ]

    # Criar uma série com índices personalizados
    serie = pd.Series(lista_dados, index=lista_indices)
    serie = pd.concat([serie, rent], axis=0)

    return serie


def ret_HPR(df_acao, df_multiplos, df_CAGR, razao_Selic, df_benchmark):
    """Calculates the expected return of assets based on EBITDA growth, DY, and multiple expansion, with penalties for low ROE and high net debt/EBITDA.

    Args:
        df_acao (pd.DataFrame): DataFrame containing fundamental data for the stock normalized by the treasury stock.
        df_multiplos (pd.DataFrame): DataFrame containing the multiples of the company (daily data).
        df_CAGR (pd.DataFrame): DataFrame containing the growth rate of fundamental data.
        razao_Selic (float): Selic ratio.
        df_benchmark (pd.DataFrame): DataFrame containing benchmark data.

    Returns:
        pd.Series: A series containing information, including sector, expected annual return, HPR, ROE, net debt/EBITDA, average DY, average CAGR, multiple expansion, current multiple, average multiple, multiple deviation, and return information.
        pd.DataFrame: A DataFrame with information about the asset.

    The function calculates the expected return of assets based on EBITDA growth, DY, and multiple expansion, with penalties for low ROE and high net debt/EBITDA.

    Example of usage:
        return_data = ret_HPR(df_acao, df_multiplos, df_CAGR, razao_Selic, sector_strategy)
    """
    ## Coletando dados iniciais
    try:
        serie_dados_iniciais = Dados_iniciais(
            df_acao,
            df_multiplos,
            df_CAGR,
            df_benchmark,
        )
    except:
        # print(f"Erro ao coletar dados iniciais de {df_acao.loc[:,'Ticker'][0]}")
        return None

    if serie_dados_iniciais is None:
        return None

    ## Descompactando a série
    [
        ticker,
        Setor,
        ML_ultm,
        ML_1std,
        M_EBITDA_1std,
        Dl_EBITDA,
        EV_SR_desvio,
        EV_EBITDA_desvio,
        CAGR_Medio_8,
        CAGR_Medio_6,
        CAGR_Medio_4,
        CAGR_Medio_2,
        CAGR_ordenado,
        CAGR_Medio_min,
        CAGR_MIN_Receita,
        DY_prev,
        DY_min,
        Ind_alta_MM,
        Ind_alta_MM_media,
        ultm_SELIC,
    ] = serie_dados_iniciais.loc[
        [
            "ticker",
            "Setor",
            "ML_ultm",
            "ML_1std",
            "M_EBITDA_1std",
            "Dl_EBITDA",
            "EV_SR_desvio",
            "EV_EBITDA_desvio",
            "CAGR_Medio_8",
            "CAGR_Medio_6",
            "CAGR_Medio_4",
            "CAGR_Medio_2",
            "CAGR_ordenado",
            "CAGR_Medio_min",
            "CAGR_MIN_Receita",
            "DY_prev",
            "DY_min",
            "Ind_alta_MM",
            "Ind_alta_MM_media",
            "ultm_SELIC",
        ]
    ]

    rent = serie_dados_iniciais.loc["rent_72_b":"rent_72_f"]

    ## Retorno esperado anual
    # HPR -> Holding period return
    # HPR = (Preço final - Preço inicial + Dividendos) / Preço inicial

    # Vetor com os dados de CAGR
    vetor_CAGR = [CAGR_Medio_8, CAGR_Medio_6, CAGR_Medio_4, CAGR_Medio_2]

    # Calcular a média dos três maiores elementos
    CAGR_max_media = np.mean(np.partition(vetor_CAGR, -3)[-3:])

    # Calcular a mediana
    CAGR_mediana = np.median(vetor_CAGR)

    # Se Cíclica seja mais conservador
    ciclica = False
    if (ML_1std <= 0) or (M_EBITDA_1std <= 0.02):
        ciclica = True

    # Cálculo do  CAGR_util e DY_util
    if ciclica:
        ### Cálculo do CAGR_util
        CAGR_util = CAGR_Medio_min
    else:
        ### Cálculo do CAGR_util

        CAGR_util = (
            CAGR_mediana * (0.3 + np.sqrt(0.7) * Ind_alta_MM - 0.4 * Ind_alta_MM**2)
            + CAGR_max_media * (np.sqrt(0.7) * Ind_alta_MM - 0.6 * Ind_alta_MM**2)
            + CAGR_Medio_min * (np.sqrt(0.7) - 1 * Ind_alta_MM) ** 2
        )

    ### HPR_inicial
    HPR_inicial = CAGR_util

    # Penalização Dívida
    lim_inf_debt = 2
    lim_sup_debt = 4
    penal_debt = Func_ajuste_reta(Dl_EBITDA, lim_inf_debt, lim_sup_debt, 1, 0.5)

    # Penalização Juros
    lim_inf_juros = 0.2
    lim_sup_juros = 0.8
    penal_juros = Func_fator_juros(razao_Selic, lim_inf_juros, lim_sup_juros, Setor)

    # Multiplicar as penalizações
    penalizacao = penal_debt * penal_juros

    # Cálculo de retorno esperado
    Retorno_anual_esperado = HPR_inicial * penalizacao

    # print(serie_penal)
    ## Lista com os returns
    lista_return = [
        Setor,
        Retorno_anual_esperado,
        HPR_inicial,
        CAGR_util,
        0,
        CAGR_ordenado,
        EV_SR_desvio,
        EV_EBITDA_desvio,
        ML_1std,
        M_EBITDA_1std,
        Ind_alta_MM,
        Ind_alta_MM_media,
        razao_Selic,
    ]

    lista_return = lista_return + (list(rent.values))

    serie_return = pd.Series(
        lista_return,
        index=[
            "Setor",
            "Retorno_anual_esperado",
            "HPR_inicial",
            "CAGR_util",
            "DY_util",
            "CAGR_ordenado",
            "EV_SR_desvio",
            "EV_EBITDA_desvio",
            "ML_1std",
            "M_EBITDA_1std",
            "Ind_alta_MM",
            "Ind_alta_MM_media",
            "razao_Selic",
            *list(rent.index),
        ],
    )

    # Série para avaliar o portfólio
    serie_info = pd.Series(
        [
            ticker,
            Ind_alta_MM_media,
            Ind_alta_MM,
            CAGR_util,
            Retorno_anual_esperado,
            HPR_inicial,
            penalizacao,
            CAGR_Medio_8,
            CAGR_Medio_6,
            CAGR_Medio_4,
            CAGR_Medio_2,
            CAGR_ordenado,
            CAGR_Medio_min,
            CAGR_MIN_Receita,
            ciclica,
            Dl_EBITDA,
            penal_debt,
            razao_Selic,
            ultm_SELIC,
            penal_juros,
        ],
        index=[
            "Ticker",
            "Ind_alta_MM_media",
            "Ind_alta_MM",
            "CAGR_util",
            "Retorno_anual_esperado",
            "HPR_inicial",
            "penalizacao",
            "CAGR_Medio_8",
            "CAGR_Medio_6",
            "CAGR_Medio_4",
            "CAGR_Medio_2",
            "CAGR_ordenado",
            "CAGR_Medio_min",
            "CAGR_MIN_Receita",
            "ciclica",
            "Dl_EBITDA",
            "penal_debt",
            "razao_Selic",
            "ultm_SELIC",
            "penal_juros",
        ],
    )

    # Criando um DataFrame com os dados
    df_info = pd.DataFrame(serie_info).T
    df_info = df_info.set_index("Ticker")

    return serie_return, df_info


def Func_info_main(df_retorno, df_cotacao_ajustado, df_benchmark):
    """Calculates the expected return of assets with different strategies depending on the sector.

    Parameters:
        df_retorno (pd.DataFrame): DataFrame containing expected return information for assets.
        df_cotacao_ajustado (pd.DataFrame): DataFrame containing adjusted stock prices.
        df_benchmark (pd.DataFrame): DataFrame containing benchmark information.

    Returns:
        info_main (list): List containing information for the main process, including the DataFrame of expected return, adjusted quotes, and sectors.
        info_verificao (list): List containing information for verification, including the return DataFrame, adjusted quotes, figures, and excluded list.

    The function calculates the expected return of assets with different strategies depending on the sector they are in.
    It also provides information for verification and essential information for the main process.

    Example of usage:
        info_main, info_verification = Func_info_main(df_retorno, df_cotacao_ajustado, df_benchmark)
    """
    ## Informações para o main
    df_retorno_new = df_retorno.sort_values(
        by="Retorno_anual_esperado",
        ascending=False,
    ).copy()
    # Extraia os primeiros quatro caracteres do índice para criar uma nova coluna 'Ticker'
    df_retorno_new["Ativo"] = df_retorno_new.index.str[:4]

    # Defina o índice como 'Ativo'
    df_retorno_new = df_retorno_new.reset_index()
    df_retorno_new = df_retorno_new.set_index("Ativo")

    # Use o método groupby para agrupar por ativo e manter a primeira ocorrência de cada grupo (com o maior retorno)
    df_retorno_new = df_retorno_new.groupby(df_retorno_new.index).first()
    df_retorno_new = df_retorno_new.set_index("index")

    # Última Selic
    ultm_SELIC = df_benchmark.loc[:, "anula100"] / 100
    ultm_SELIC = ultm_SELIC.dropna()
    ultm_SELIC = ultm_SELIC.loc[ultm_SELIC.index.max()]

    # Retirar ativos com rentabilidade esperada menor que SELIC
    df_retorno_new = df_retorno_new[
        df_retorno_new["Retorno_anual_esperado"] >= ultm_SELIC
    ]
    df_retorno_new = df_retorno_new.sort_index()

    # Redefinir o df_cotacao_ajustado
    ativos = df_retorno_new.index
    df_cot_new = df_cotacao_ajustado.loc[:, ativos].copy()

    ## Filtrar os últimos 3 anos
    df_cot_new = df_cot_new.iloc[-3 * 252 :, :]  # últimos 3 anos

    ## Retirar os valores nan
    # Se tiver pouco nan não retirar
    contagem_nan = df_cot_new.isna().sum(axis=0)
    colunas_a_remover = contagem_nan[
        contagem_nan > 5
    ].index  # Remover coluna se tiver mais de 5 nan
    df_cot_new = df_cot_new.drop(columns=colunas_a_remover)

    # Verificando se as últimas 3 linhas tem Nan
    df_ultimas = df_cot_new.copy().tail(3)
    nao_tem_null = []
    for coluna in df_ultimas.columns:
        df_ultimas_2 = df_ultimas[coluna]
        tem_null = df_ultimas_2.isnull().any()
        if tem_null == False:
            nao_tem_null.append(coluna)

    # Selecionando as colunas adequadas
    df_cot_new = df_cot_new.loc[:, nao_tem_null]

    # Se tiver muito nan retirar a linha
    df_cot_new = df_cot_new.dropna(how="any", axis=0)

    ## Retirar os ativos que não estão no df_cot_new
    ativos = df_cot_new.columns
    df_retorno_new = df_retorno_new.loc[df_retorno_new.index.isin(ativos), :]

    ## Definir as informações que serão utilizadas para a montagem do portfólio
    serie_retornos = df_retorno_new["Retorno_anual_esperado"]

    # Adicionando o ativo da SELIC no retorno
    serie_retornos.loc["LFTS3"] = ultm_SELIC

    # Adicionando o ativo da SELIC nas cotações
    df_cot_atu = pd.concat([df_cot_new, df_benchmark["LFTS3"]], axis=1)
    df_cot_atu = df_cot_atu.dropna(how="any", axis=0)

    # Adicionando o ativo da SELIC no setor
    serie_setor = df_retorno_new["Setor"]
    serie_setor.loc["LFTS3"] = "RF"

    info_main = [serie_retornos, df_cot_atu, serie_setor]
    return info_main


# %%
def Func_info_verificacao(
    df_retorno_v,
    df_cotacao_v,
    df_benchmark_v,
    df_info_v,
    ordenar_ret=True,
):
    """Generates information for verification and study based on the provided data.

    Parameters:
        df_retorno_v (pd.DataFrame): DataFrame containing return information for verification.
        df_cotacao_v (pd.DataFrame): DataFrame containing adjusted stock prices for verification.
        df_benchmark_v (pd.DataFrame): DataFrame containing benchmark information for verification.
        df_info_v (pd.DataFrame): DataFrame containing additional information for verification.
        ordenar_ret (bool, optional): Whether to sort the DataFrame by expected return. Default is True.

    Returns:
        list: List containing DataFrames for verification and study.

    The function generates information for verification and study based on the provided data.

    Example of usage:
        info_verificacao = Func_info_verificacao(df_retorno_v, df_cotacao_v, df_benchmark_v, df_info_v, ordenar_ret=True)
    """
    ## Informações para verificação
    # Retirar os valores nan
    df_cotacao_ajustado_v = df_cotacao_v.copy()
    df_cotacao_ajustado_v = df_cotacao_ajustado_v.dropna(how="all", axis=0)
    df_cotacao_ajustado_v = df_cotacao_ajustado_v.dropna(how="all", axis=1)
    ativos = df_cotacao_ajustado_v.columns

    # Retirar os ativos que foram excluídos
    df_retorno_v = df_retorno_v.loc[df_retorno_v.index.isin(ativos), :]

    # Incluindo o benchmark
    df_cotacao_bench_v = pd.concat(
        [df_cotacao_ajustado_v, df_benchmark_v.loc[:, ["LFTS3", "IBOV"]]],
        axis=1,
    )

    # Ordenar por retorno esperado
    if ordenar_ret:
        df_retorno_v = df_retorno_v.sort_values(
            by="Retorno_anual_esperado",
            ascending=False,
        )

    ## Organizando os dados para a verificação/estudo
    df_info = df_retorno_v.loc[:, ["Retorno_anual_esperado", "Setor"]]
    df_info = df_info.rename(columns={"Retorno_anual_esperado": "Ret_anual"})
    df_info = pd.concat([df_info, df_info_v], axis=1)
    df_info = df_info.loc[
        :,
        [
            "Setor",
            "Ret_anual",
            "Ind_alta_MM_media",
            "Ind_alta_MM",
            "CAGR_util",
            "Retorno_anual_esperado",
            "HPR_inicial",
            "penalizacao",
            "CAGR_Medio_8",
            "CAGR_Medio_6",
            "CAGR_Medio_4",
            "CAGR_Medio_2",
            "CAGR_ordenado",
            "CAGR_Medio_min",
            "CAGR_MIN_Receita",
            "ciclica",
            "Dl_EBITDA",
            "penal_debt",
            "razao_Selic",
            "ultm_SELIC",
            "penal_juros",
        ],
    ]

    ## Data da última cotação
    ultm_data = df_cotacao_ajustado_v.index.max()

    # df_info
    df_info.index.name = "Ticker"
    df_info.loc[:, "Data_Simulacao"] = ultm_data
    df_info = df_info.reset_index()
    df_info = df_info.set_index("Data_Simulacao")

    return [
        df_info,
        df_retorno_v.copy(),
        df_cotacao_ajustado_v.copy(),
        df_cotacao_bench_v,
    ]


# %%
def main_ret(dict_df_acoes, razao_Selic, df_benchmark):
    """Calculates the expected return of assets with different strategies depending on the sector.

    Parameters:
        dict_df_acoes (dict): Dictionary with keys being the assets and values containing (df_acao, df_multiplos, df_CAGR).
        razao_Selic (float): Float containing the ratio between the expectation of SELIC and the current value of the forecast.
        df_benchmark (pd.DataFrame): DataFrame containing benchmark information.

    Returns:
        tuple: Tuple containing information for the main process and verification.

    The function calculates the expected return of assets with different strategies depending on the sector they are in.
    It also provides information for verification and essential information for the main process.

    Example of usage:
        info_main, info_verificao = main_ret(dict_df_acoes, razao_Selic, df_benchmark)
    """
    ## Lista de ativos elegíveis
    lista_ativos_elegiveis = list(dict_df_acoes.keys())
    lista_ativos_elegiveis.sort()

    ## Criando o dataframe de retorno
    df_retorno = pd.DataFrame(
        columns=[
            "Setor",
            "Retorno_anual_esperado",
            "HPR_inicial",
            "CAGR_util",
            "DY_util",
            "CAGR_ordenado",
            "EV_SR_desvio",
            "EV_EBITDA_desvio",
            "ML_1std",
            "M_EBITDA_1std",
            "Ind_alta_MM",
            "Ind_alta_MM_media",
            "razao_Selic",
            "rent_72_b",
            "rent_36_b",
            "rent_12_b",
            "rent_6_b",
            "rent_3_b",
            "rent_1_b",
            "rent_1_f",
            "rent_3_f",
            "rent_6_f",
            "rent_12_f",
            "rent_36_f",
            "rent_72_f",
        ],
    )
    # Criando o dataframe de cotação ajustada
    df_cotacao_ajustado = pd.DataFrame()
    # Criando o dataframe de penalidades
    df_info = pd.DataFrame()

    # Lista de ativos excluídos
    lista_excluidos = []

    for ativo in lista_ativos_elegiveis:
        ## Pegando os dados dos ativos
        [df_acao, df_multiplos, df_CAGR] = dict_df_acoes[ativo]

        try:
            # Série com as informações do setor
            lista_return, df_info_ativo = ret_HPR(
                df_acao,
                df_multiplos,
                df_CAGR,
                razao_Selic,
                df_benchmark,
            )

            # Se o retorno for None, o ativo será excluído
            if lista_return is None:
                lista_excluidos.append(ativo)
                continue

            ## Adicionando o retorno esperado
            df_retorno.loc[ativo, :] = lista_return

            ## Adicionando a cotação ajustada
            df_ativo = df_multiplos[["Fech_Ajustado"]].copy()
            df_ativo.columns = [ativo]
            df_cotacao_ajustado = pd.concat(
                [df_cotacao_ajustado, df_ativo],
                axis=1,
                join="outer",
            )

            ## Adicionando as informações
            df_info = pd.concat([df_info, df_info_ativo], axis=0, join="outer")

        except TypeError:
            # print(f"Ocorreu um TypeError: {e} em {ativo}.")
            pass
        except:
            # print(f"Ticker: {ativo} foi excluído.")
            # print()
            pass

    ## Informações para o main
    info_main = Func_info_main(df_retorno, df_cotacao_ajustado, df_benchmark)

    ## Informações para verificação
    info_verificao = Func_info_verificacao(
        df_retorno,
        df_cotacao_ajustado,
        df_benchmark,
        df_info,
    )

    return info_main, info_verificao


# %% [markdown]
# ### Funções para construção do portfólio


# %%
def Func_dados_inicias(info_main):
    """Collects initial data for portfolio construction.

    Parameters:
        info_main (list): List of essential information, including the DataFrame of expected returns, adjusted stock prices, and sectors.

    Returns:
        pd.Series: Series with the expected returns of assets.
        pd.DataFrame: DataFrame with the adjusted stock prices of assets.
        pd.Series: Series with information about the sector of each asset.
        float: Risk-free rate, usually associated with SELIC.
        pd.Series: Series with the most recent prices of assets.

    The function collects initial information necessary for portfolio construction, including expected returns, adjusted stock prices, information about sectors, the risk-free rate, and the most recent prices.

    Example of usage:
        expec_retorno, cotacoes, setores, taxa_livre_risco, latest_prices = Func_dados_inicias(info_main)
    """
    ## Dados de retorno esperados e cotação ajustada
    expec_retorno = info_main[0]
    cotacoes = info_main[1]
    setores = info_main[2]

    # prices as of the day you are allocating
    latest_prices = cotacoes.iloc[-1]

    # Obtendo a taxa livre de risco
    taxa_livre_risco = expec_retorno["LFTS3"]

    ## Constantes

    return expec_retorno, cotacoes, setores, taxa_livre_risco, latest_prices


# %%
def Func_alocacao_setorial(setores, maior_per_setor=0.20, min_RF=0.2):
    """Builds portfolio allocation constraints based on sectors.

    Parameters:
        setores (pd.Series): Series containing information about the sector of each asset.
        maior_per_ativo (int, optional): Maximum percentage allocation for each asset. Default is 5%.
        maior_per_setor (int, optional): Maximum percentage allocation for each sector. Default is 20%.
        min_RF (float, optional): Minimum percentage allocation in Fixed Income. Default is 0.2 (20%).

    Returns:
        dict: Dictionary with sectors and their corresponding allocations.
        dict: Dictionary with minimum allocation values per sector.
        dict: Dictionary with maximum allocation values per sector.

    The function builds portfolio allocation constraints based on the sectors of the assets. It sets the maximum percentage allocation allowed per asset and per sector, as well as the minimum percentage allocation in Fixed Income.

    Example of usage:
        setores_dic, sector_lower, sector_upper = Func_alocacao_setorial(setores)
    """
    ## Criando as restrições de mínimo
    unique_sectors = list(set(setores))  # Obtém os valores únicos usando set()
    setores_dic = setores.to_dict()
    sector_lower = {setor: 0 for setor in unique_sectors}
    ## Alocação mínima em RF
    sector_lower["RF"] = min_RF

    # Criar 'sector_upper' com o máximo para todos os setores
    sector_upper = {setor: maior_per_setor for setor in unique_sectors}
    sector_upper["RF"] = 1

    return setores_dic, sector_lower, sector_upper


def Func_ajuste_Pesos(
    expec_retorno,
    S,
    setores_dic,
    sector_lower,
    sector_upper,
    maior_per_ativo,
    taxa_livre_risco,
    latest_prices,
    gamma_tun=5,
    menor_per_aloc=0.005,
):
    """Adjusts portfolio allocation limits based on sectors.

    Parameters:
        expec_retorno (pd.Series): Series containing expected returns for each asset.
        S (pd.DataFrame): Covariance matrix of asset returns.
        setores_dic (dict): Dictionary with sectors and their corresponding allocations.
        sector_lower (dict): Dictionary with minimum allocation values per sector.
        sector_upper (dict): Dictionary with maximum allocation values per sector.
        maior_per_ativo (int, optional): Maximum percentage allocation for each asset. Default is 5%.
        taxa_livre_risco (float): Risk-free rate.
        latest_prices (pd.Series): Series containing the latest prices of assets.
        gamma_tun (float, optional): Tuning parameter for L2 regularization. Default is 5.
        menor_per_aloc (float, optional): Minimum percentage allocation. Default is 0.005.

    Returns:
        weights (pd.Series): Series with asset weights in the optimized portfolio.
        performance (tuple): Portfolio performance metrics (Expected annual return, Annual volatility, Sharpe Ratio).
        weights_sem_nulos (OrderedDict): Ordered dictionary excluding items with zero weights.
        latest_prices_n (pd.Series): Series containing the latest prices of assets included in the optimized portfolio.

    The function adjusts portfolio allocation limits based on asset sectors. It defines the maximum percentage allocation allowed per asset and per sector, as well as the minimum percentage allocation in Fixed Income.

    Example of usage:
        weights, performance, weights_sem_nulos, latest_prices_n = Func_ajuste_Pesos(expec_retorno, S, setores_dic, sector_lower, sector_upper, maior_per_ativo, taxa_livre_risco, latest_prices)
    """
    # Cria o objeto EfficientFrontier
    ef = EfficientFrontier(
        expec_retorno,
        S,
        weight_bounds=(0, 1),
    )  # weight_bounds automatically set to (0, 1)

    # Restrição Setorial
    ef.add_sector_constraints(setores_dic, sector_lower, sector_upper)

    ## Adicionando restrições por ativo
    lista_ativos = list(expec_retorno.index)
    with contextlib.suppress(Exception):
        lista_ativos.remove("LFTS3")

    for ativo in lista_ativos:
        index = ef.tickers.index(ativo)
        ef.add_constraint(lambda w: w[index] <= maior_per_ativo)

    ef.add_objective(
        objective_functions.L2_reg,
        gamma=gamma_tun,
    )  # gamma is the tuning parameter

    # Diminuir com pouco a taxa mínimo de risco porque tem que ter pelo menos
    # um ativo com retorno maior do que esta taxa, sendo assim, fiz esta alteração
    taxa_livre_risco = taxa_livre_risco * 0.9

    # Máximar o Sharp
    try:
        ef.max_sharpe(risk_free_rate=taxa_livre_risco)
        weights = ef.clean_weights(cutoff=menor_per_aloc)
        performance = ef.portfolio_performance(
            verbose=False,
            risk_free_rate=taxa_livre_risco,
        )
    except:
        print("Cálculo do Sharpe do pacote de Estrategia_retorno_my")
        raise
    ## performance = (Expected annual return, Annual volatility, Sharpe Ratio)

    # Crie um novo OrderedDict excluindo os itens com valores nulos
    weights_sem_nulos = OrderedDict(
        (chave, valor) for chave, valor in weights.items() if valor != 0.0
    )

    # Últimos preços dos ativos
    ativos = list(weights_sem_nulos.keys())
    latest_prices_n = latest_prices[ativos]

    return weights, performance, weights_sem_nulos, latest_prices_n


def Func_alocacao_dia(
    info_main,
    Montante_Total,
    maior_per_ativo=0.1,
    maior_per_setor=0.20,
    min_RF=0.2,
    gamma_tun=5,
    menor_per_aloc=0.005,
):
    """Calculates the daily asset allocation in a portfolio based on sector, risk, and other constraints.

    Parameters:
        Montante_Total (float): The total amount available for investment.
        info_main (list): List containing information for the main process, including expected returns, adjusted prices, and sectors.
        maior_per_ativo (float, optional): The maximum allowed allocation per asset in percentage. Default is 5%.
        maior_per_setor (float, optional): The maximum allowed allocation per sector in percentage. Default is 20%.
        min_RF (float, optional): The minimum allowed allocation in fixed income in percentage. Default is 0.2%.
        gamma_tun (float, optional): Penalty parameter for the distribution of zero weights. Default is 5.
        menor_per_aloc (float, optional): Minimum percentage allocation to keep an asset in the portfolio. Default is 0.5%.

    Returns:
        pd.DataFrame: DataFrame containing the money allocation in each asset on a given day. Columns represent assets, and rows represent dates.

    The function calculates the daily asset allocation in a portfolio based on various constraints, including maximum allocation per asset, maximum allocation per sector, and minimum allocation in fixed income. It also allows specifying a percentage of short in the portfolio and applies penalties to avoid zero weights on assets. The result is a DataFrame that shows the money allocation in each asset on specific dates.

    Example of usage:
        df_alloc_money, performance, weights_sem_nulos = Func_alocacao_dia(Montante_Total, info_main)
    """
    ## Dados de retorno esperados e cotação ajustada
    (
        expec_retorno,
        cotacoes,
        setores,
        taxa_livre_risco,
        latest_prices,
    ) = Func_dados_inicias(info_main)

    ## Constantes
    S = risk_models.CovarianceShrinkage(cotacoes).ledoit_wolf()
    data_alocacao = cotacoes.index.max()

    ## Setores
    setores_dic, sector_lower, sector_upper = Func_alocacao_setorial(
        setores,
        maior_per_setor,
        min_RF,
    )

    ## Ajuste dos pesos
    try:
        weights, performance, weights_sem_nulos, latest_prices_n = Func_ajuste_Pesos(
            expec_retorno,
            S,
            setores_dic,
            sector_lower,
            sector_upper,
            maior_per_ativo,
            taxa_livre_risco,
            latest_prices,
            gamma_tun,
            menor_per_aloc,
        )
    except:
        print(
            "Func_ajuste_Pesos dentro de Func_alocacao_dia do pacote de Estrategia_retorno_my",
        )
        raise

    ## Montagem da Carteira
    da = DiscreteAllocation(weights_sem_nulos, latest_prices_n, Montante_Total)
    try:
        alloc, leftover = da.greedy_portfolio()
    except:
        print(weights_sem_nulos, latest_prices_n, Montante_Total)
        raise
        # alloc, leftover = da.greedy_portfolio()
    # print(f"Portólio alocado com R$ {leftover:.2f} restantes")
    alloc_money = {
        key: round(value * latest_prices[key], 2)
        for key, value in alloc.items()
        if value != 0
    }

    # Transformando o dicionário em um DataFrame com as colunas definidas pelos tickers
    df_alloc_money = pd.DataFrame([alloc_money], columns=alloc_money.keys())

    # Definindo a data como índice (neste exemplo, usando '2023-01-01' como data)
    df_alloc_money["Data"] = data_alocacao
    df_alloc_money = df_alloc_money.set_index("Data")

    df_alloc_money.loc[data_alocacao, "leftover"] = leftover
    df_alloc_money.loc[data_alocacao, "Total"] = df_alloc_money.loc[
        data_alocacao,
        :,
    ].sum(axis=0)

    return df_alloc_money, performance, weights_sem_nulos


# %%
def Func_construir_portfolio(
    data_simulacao,
    Capital,
    maior_per_ativo=0.1,
    maior_per_setor=0.20,
    min_RF=0.2,
    gamma_tun=5,
    menor_per_aloc=0.005,
    IBOV=True,
):
    """Builds an investment portfolio based on various constraints, including expected returns.

    Parameters:
        Capital (float): The total amount available for investment.
        maior_per_ativo (float, optional): The maximum allowed allocation per asset in percentage. Default is 5%.
        maior_per_setor (float, optional): The maximum allowed allocation per sector in percentage. Default is 20%.
        min_RF (float, optional): The minimum allowed allocation in fixed income in percentage. Default is 0.2%.
        gamma_tun (float, optional): Penalty parameter for the distribution of zero weights. Default is 5.
        menor_per_aloc (float, optional): Minimum percentage allocation to keep an asset in the portfolio. Default is 0.5%.
        IBOV (bool, optional): Whether to include IBOV (Brazilian stock market index) in the analysis. Default is True.

    Returns:
        df_alloc_money (pd.DataFrame): DataFrame containing the money allocation in each asset on specific dates.

    The function builds an investment portfolio based on various constraints, including expected returns. It loads data with the simulation date, calculates the expected returns of assets, applies constraints based on sectors and other metrics, and provides the money allocation in each asset on specific dates.

    Example of usage:
        df_alloc_money = Func_construir_portfolio(data_simulacao, Capital, maior_per_ativo, maior_per_setor, min_RF, gamma_tun, menor_per_aloc)
    """
    # Carregar os dados necessários com a data de simulação
    dict_df_acoes, df_benchmark, df_Expectativa_Selic = load_data_main(
        data_simulacao,
        IBOV,
    )

    ## Obter a previsão da expectativa da SELIC
    Indicacoes_Alta_selic = df_Expectativa_Selic.loc[
        df_Expectativa_Selic.index.max(),
        "Indicacoes_Alta",
    ]

    ## Calcular a rentabilidade esperada dos ativos
    info_main, info_verificao = main_ret(
        dict_df_acoes,
        Indicacoes_Alta_selic,
        df_benchmark,
    )

    df_alloc_money, performance, weights_sem_nulos = Func_alocacao_dia(
        info_main,
        Capital,
        maior_per_ativo,
        maior_per_setor,
        min_RF,
        gamma_tun,
        menor_per_aloc,
    )

    return df_alloc_money, info_verificao
