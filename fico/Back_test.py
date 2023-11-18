""" Este módulo faz o Backtesting da estratégia.

   """

import pandas as pd
from fico.load_data import load_data
from fico.Func_construir_portfolio import Func_construir_portfolio
from typing import Tuple


def Func_Portfolio_intervalo(
    data_inicial: str,
    data_final: str,
    Capital_Inicial: float,
    df_benchmark: pd.DataFrame,
    maior_per_ativo: float = 0.1,
    maior_per_setor: float = 0.2,
    min_RF: float = 0.2,
    gamma_tun: int = 5,
    menor_per_aloc: float = 0.005,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constrói e rastreia o desempenho de um portfólio ao longo de um intervalo de datas especificado.

    Args:
        data_inicial (str): Data inicial para a construção do portfólio no formato 'YYYY-MM-DD'.
        data_final (str): Data final para a construção do portfólio no formato 'YYYY-MM-DD'.
        Capital_Inicial (float): O montante inicial de capital para investir no portfólio.
        df_benchmark (pd.DataFrame): DataFrame contendo os dados de referência do mercado.
        maior_per_ativo (float): A alocação máxima permitida para um único ativo no portfólio, padrão é 0.1 (10%).
        maior_per_setor (float): A alocação máxima permitida para um setor específico no portfólio, padrão é 0.2 (20%).
        min_RF (float): A alocação mínima em ativos de renda fixa no portfólio, padrão é 0.2 (20%).
        gamma_tun (int): O parâmetro de ajuste que penaliza a distribuição de pesos nulos, padrão é 5.
        menor_per_aloc (float): O valor mínimo de alocação para um ativo no portfólio, padrão é 0.005 (0.5%).

    Returns:
        df_portifolio (pd.DataFrame): DataFrame que rastreia o desempenho do portfólio ao longo do intervalo de datas especificado.
        info_verificao (pd.DataFrame): DataFrame com dados sobre os ativos dos portfólio.

    Notes:
        A função constrói um portfólio com base nos parâmetros especificados e rastreia seu desempenho diariamente ao longo do intervalo de datas fornecido.

    Examples:
        >>> df_portifolio, info_verificao = Func_Portfolio_intervalo('2023-01-01', '2023-12-31', 1000000, df_benchmark)
    """
    # Contruindo o portifolio
    try:
        df_alloc_money, info_verificao = Func_construir_portfolio(
            data_inicial,
            Capital_Inicial,
            maior_per_ativo,
            maior_per_setor,
            min_RF,
            gamma_tun,
            menor_per_aloc,
        )
    except:
        print("Func_construir_portfolio", data_inicial)
        raise
    ativos_portifolio = df_alloc_money.columns
    ativos_portifolio = ativos_portifolio.drop(["leftover", "Total"])
    leftover = df_alloc_money.loc[data_inicial, "leftover"]

    # Cotações
    todas_cotacoes = df_benchmark.copy()
    condicao_data = (todas_cotacoes.index >= data_inicial) & (
        todas_cotacoes.index <= data_final
    )
    todas_cotacoes = todas_cotacoes.loc[condicao_data]

    # Dataframe com os retornos
    df_retorno = todas_cotacoes.pct_change()
    df_retorno_acumulado = (1 + df_retorno).cumprod()
    df_retorno_acumulado.iloc[0, :] = 1

    # Indices de referencia
    indices_rebalanco = df_retorno_acumulado.index

    # Dataframe com os retornos dos ativos do portifolio
    df_retorno_acumulado_port = df_retorno_acumulado.loc[:, ativos_portifolio]

    # Alocação do portifolio
    df_money_port = df_alloc_money.loc[:, ativos_portifolio]
    grana = df_money_port.iloc[0, :].values
    df_money_port = pd.DataFrame(index=indices_rebalanco, columns=ativos_portifolio)
    df_money_port.loc[:] = grana

    # Dataframe da posição do portifolio em dinheiro e diariamente
    df_portifolio = df_retorno_acumulado_port * df_money_port
    df_portifolio["Montante"] = df_portifolio.sum(axis=1) + leftover

    return df_portifolio, info_verificao


def Func_retorno_carteira(
    data_simulacao_inicial,
    data_simulacao_final,
    Capital_Inicial,
    dias_rebalancear,
    maior_per_ativo=0.1,
    maior_per_setor=0.2,
    min_RF=0.2,
    gamma_tun=5,
    menor_per_aloc=0.005,
):
    """Calculates the portfolio return over a specific period based on rebalancing.

    Parameters:
        data_simulacao_inicial (str): Initial date for the simulation in the 'YYYY-MM-DD' format.
        data_simulacao_final (str): Final date for the simulation in the 'YYYY-MM-DD' format.
        Capital_Inicial (float): The initial capital amount to invest in the portfolio.
        dias_rebalancear (int): The number of days between portfolio rebalances.
        maior_per_ativo (float): Maximum allowed allocation for a single asset in the portfolio, default is 0.1 (10%).
        maior_per_setor (float): Maximum allowed allocation for a specific sector in the portfolio, default is 0.2 (20%).
        min_RF (float): Minimum allocation in fixed-income assets in the portfolio, default is 0.2 (20%).
        gamma_tun (int): Tuning parameter penalizing the distribution of null weights, default is 5.
        menor_per_aloc (float): Minimum allocation value for an asset in the portfolio, default is 0.005 (0.5%).

    Returns:

        retorno_carteira (pd.Series): Series of portfolio returns over the simulation period.
        serie_IBOV (pd.Series): Series of IBOV index returns aligned with the portfolio returns series.
        serie_LFTS3 (pd.Series): Series of SELIC returns aligned with the portfolio returns series.
        df_portifolio_total (pd.DataFrame): DataFrame recording portfolios during rebalancing.
        df_benchmark (pd.DataFrame): DataFrame with benchmark data.
        indices_benchmark (list): Dates of rebalances.
        df_exportar (pd.DataFrame): DataFrame recording data for export.
        info_verificao_total (list): List of DataFrames with data for export.

    Notes:
        The function performs a portfolio simulation based on the provided parameters, rebalancing the portfolio at specific intervals, and calculating the returns. It also provides market reference information aligned with the portfolio returns.

    Example:
        >>> retorno_carteira, serie_IBOV, serie_LFTS3, df_portifolio_total,
        df_benchmark, indices_benchmark, df_exportar, info_verificao_total  = Func_retorno_carteira('2023-01-01', '2023-12-31', 1000000, 7)
    """
    ## Carregar dados
    [dict_df_acoes, df_benchmark, df_Expectativa_Selic_mensal] = load_data(
        data_simulacao_final,
    )

    ## Dados do Benchmark
    df_benchmark = df_benchmark.dropna(subset=["LFTS3"])
    df_benchmark = df_benchmark.drop(columns=["anula100", "diario"])
    df_benchmark = df_benchmark.loc[data_simulacao_inicial:data_simulacao_final]
    retorno_benchmark = df_benchmark.pct_change()
    retorno_benchmark_IBOV = retorno_benchmark["IBOV"].dropna()
    retorno_benchmark_LFST3 = retorno_benchmark["LFTS3"].dropna()
    indices_benchmark = df_benchmark.index
    indices_benchmark = indices_benchmark[::dias_rebalancear]
    indices_benchmark = [*list(indices_benchmark), df_benchmark.index[-1]]
    data_simulacao_inicial = indices_benchmark[0]
    ## (FIM) Dados do Benchmark

    ## Construir portifolio
    for i in range(len(indices_benchmark) - 1):
        data_inicial = indices_benchmark[i]
        data_final = indices_benchmark[i + 1]
        try:
            df_portifolio, info_verificao = Func_Portfolio_intervalo(
                data_inicial,
                data_final,
                Capital_Inicial,
                df_benchmark,
                maior_per_ativo,
                maior_per_setor,
                min_RF,
                gamma_tun,
                menor_per_aloc,
            )
        except:
            print("Func_Portfolio_intervalo", data_inicial, data_final)
            raise
        Capital_Inicial = df_portifolio["Montante"].iloc[-1]
        if i == 0:
            df_portifolio_total = df_portifolio.copy()
            df_exportar = df_portifolio.copy()
        else:
            df_portifolio_total = pd.concat([df_portifolio_total[:-1], df_portifolio])
            df_exportar = pd.concat([df_exportar, df_portifolio])

        if i == 0:
            info_verificao_total = info_verificao.copy()
        else:
            for tamanho in range(len(info_verificao_total)):
                try:
                    info_verificao_total[tamanho] = pd.concat(
                        [info_verificao_total[tamanho], info_verificao[tamanho]],
                        axis=0,
                    )
                except:
                    print(data_inicial, tamanho)
                    raise

        ## (FIM) Construir portifolio

    # Retorno da carteira
    df_portifolio_total["retorno"] = df_portifolio_total["Montante"].pct_change()
    retorno_carteira = df_portifolio_total["retorno"].dropna()
    serie_IBOV = retorno_benchmark_IBOV[retorno_carteira.index]
    serie_LFTS3 = retorno_benchmark_LFST3[retorno_carteira.index]

    return (
        retorno_carteira,
        serie_IBOV,
        serie_LFTS3,
        df_portifolio_total,
        df_benchmark,
        indices_benchmark,
        df_exportar,
        info_verificao_total,
    )
