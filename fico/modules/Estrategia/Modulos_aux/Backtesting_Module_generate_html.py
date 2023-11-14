import pandas as pd
import quantstats as qs

""" Geração do HTML do backtest"""


def Func_ret_aa_serie(serie):
    """Calcula o retorno anual de uma série de preços.

    A função recebe uma série de preços e calcula o retorno anual percentual
    composto. A série é esperada como um objeto que possui um índice ordenado
    cronologicamente.

    Args:
        serie (pandas.Series): Uma série de preços ordenados cronologicamente.

    Returns:
        float: O retorno anual percentual composto da série.


    Example:
        >>> prices = pd.Series([100, 105, 98, 110, 120])
        >>> Func_ret_aa_serie(prices)
        4.76
    """
    # Ordena a série pelo índice, garantindo ordem cronológica
    serie = serie.sort_index()

    # Calcula o número de anos na série (assumindo 252 dias de negociação por ano)
    anos = len(serie) / 252

    # Calcula o retorno anual percentual composto da série
    ret_serie = (serie + 1).cumprod()[-1]
    ret_serie_aa_pct = round(((ret_serie ** (1 / anos)) - 1) * 100, 2)

    return ret_serie_aa_pct


def Func_ret_aa_varios_periodos(serie):
    """Calcula o retorno anual de uma série de preços para diferentes períodos.

    A função recebe uma série de preços e calcula o retorno anual percentual composto
    para os períodos totais, últimos 3 anos, últimos 5 anos e últimos 10 anos.

    Args:
        serie (pandas.Series): Uma série de preços ordenados cronologicamente.

    Returns:
        tuple: Uma tupla contendo os retornos anuais percentuais composto para os
        períodos total, 3 anos, 5 anos e 10 anos, respectivamente.



    Example:
        >>> prices = pd.Series([100, 105, 98, 110, 120, 130, 125, 140, 150, 160])
        >>> Func_ret_aa_varios_periodos(prices)
        (60.0, 21.78, 44.58, 110.36)
    """
    # Calcula o retorno anual total da série
    ret_serie_aa_pct_total = Func_ret_aa_serie(serie)

    # Obtém a data do último dia na série
    data_ultimo_dia = serie.index[-1]

    # Calcula o retorno anual para os últimos 3 anos
    serie_3anos = serie.loc[serie.index >= data_ultimo_dia - pd.DateOffset(years=3)]
    ret_serie_aa_pct_3anos = Func_ret_aa_serie(serie_3anos)

    # Calcula o retorno anual para os últimos 5 anos
    serie_5anos = serie.loc[serie.index >= data_ultimo_dia - pd.DateOffset(years=5)]
    ret_serie_aa_pct_5anos = Func_ret_aa_serie(serie_5anos)

    # Calcula o retorno anual para os últimos 10 anos
    serie_10anos = serie.loc[serie.index >= data_ultimo_dia - pd.DateOffset(years=10)]
    ret_serie_aa_pct_10anos = Func_ret_aa_serie(serie_10anos)

    return (
        ret_serie_aa_pct_total,
        ret_serie_aa_pct_3anos,
        ret_serie_aa_pct_5anos,
        ret_serie_aa_pct_10anos,
    )


def Func_relatorio_html(
    retorno_carteira,
    benchmark,
    benchmark_titulo,
    nome_arquivo,
    titulo,
):
    """Gera um relatório HTML com correção do CAGR e informações de retorno anual.

    A função utiliza a biblioteca `qs.reports.html` para gerar um relatório HTML
    comparando o retorno da carteira com um benchmark. Após a geração do HTML,
    ela corrige as informações de CAGR no HTML gerado com os retornos anuais
    corretos para diferentes períodos (3 anos, 5 anos, 10 anos e all-time).

    Args:
        retorno_carteira (pandas.Series): Série de retorno da carteira.
        benchmark (pandas.Series): Série de retorno do benchmark.
        benchmark_titulo (str): Título/nome do benchmark utilizado.
        nome_arquivo (str): Nome do arquivo HTML a ser gerado.
        titulo (str): Título do relatório.

    Returns:
        None


    Example:
        >>> portfolio_returns = pd.Series([0.02, 0.03, -0.01, 0.02, 0.01])
        >>> benchmark_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.008])
        >>> Func_relatorio_html(portfolio_returns, benchmark_returns, "S&P 500", "relatorio.html", "Relatório de Desempenho")
        Relatório HTML gerado com sucesso!
    """
    # Gerar Relatório HTML
    qs.reports.html(
        retorno_carteira,
        benchmark=benchmark,
        benchmark_title=benchmark_titulo,
        output=nome_arquivo,
        title=titulo,
    )

    # Consertando o CAGR
    # Calculando os retorno anual correto
    ret_aa_IBOV = Func_ret_aa_varios_periodos(benchmark)
    ret_aa_carteira = Func_ret_aa_varios_periodos(retorno_carteira)

    # Alterando o HTML
    import re  # Importe a biblioteca re

    # Abra o arquivo HTML
    with open(nome_arquivo, encoding="utf-8") as file:
        html = file.read()

    # Substituindo as expressões
    try:
        # Use expressões regulares para encontrar a tag <tr> com os valores de CAGR%
        pattern = r"<tr><td>CAGR﹪</td><td>[\d.]+%</td><td>[\d.]+%</td></tr>"
        # Substitua a correspondência pela nova tag com os valores desejados
        replacement = f"<tr><td>CAGR﹪</td><td>{ret_aa_IBOV[0]}%</td><td>{ret_aa_carteira[0]}%</td>"
        html = re.sub(pattern, replacement, html)

        # Use expressões regulares para encontrar a tag <tr>
        pattern = r"<tr><td>3Y \(ann.\)</td><td>[\d.]+%</td><td>[\d.]+%</td></tr>"
        # Substitua a correspondência pela nova tag com os valores desejados
        replacement = f"<tr><td>3Y (ann.)</td><td>{ret_aa_IBOV[1]}%</td><td>{ret_aa_carteira[1]}%</td></tr>"
        html = re.sub(pattern, replacement, html)

        # Use expressões regulares para encontrar a tag <tr>
        pattern = r"<tr><td>5Y \(ann.\)</td><td>[\d.]+%</td><td>[\d.]+%</td></tr>"
        # Substitua a correspondência pela nova tag com os valores desejados
        replacement = f"<tr><td>5Y (ann.)</td><td>{ret_aa_IBOV[2]}%</td><td>{ret_aa_carteira[2]}%</td></tr>"
        html = re.sub(pattern, replacement, html)

        # Use expressões regulares para encontrar a tag <tr>
        pattern = r"<tr><td>10Y \(ann.\)</td><td>[\d.]+%</td><td>[\d.]+%</td></tr>"
        # Substitua a correspondência pela nova tag com os valores desejados
        replacement = f"<tr><td>10Y (ann.)</td><td>{ret_aa_IBOV[3]}%</td><td>{ret_aa_carteira[3]}%</td></tr>"
        html = re.sub(pattern, replacement, html)

        # Use expressões regulares para encontrar a tag <tr>
        pattern = r"<tr><td>All-time \(ann.\)</td><td>[\d.]+%</td><td>[\d.]+%</td></tr>"
        # Substitua a correspondência pela nova tag com os valores desejados
        replacement = f"<tr><td>All-time (ann.)</td><td>{ret_aa_IBOV[0]}%</td><td>{ret_aa_carteira[0]}%</td></tr>"
        html = re.sub(pattern, replacement, html)
    except:
        pass

    # Salve o HTML modificado no arquivo original
    with open(nome_arquivo, "w", encoding="utf-8") as file:
        file.write(html)

    print("Relatório HTML gerado com sucesso!")
