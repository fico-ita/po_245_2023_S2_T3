""" Geração do HTML do backtest"""

import pandas as pd
import quantstats as qs


def Func_ret_aa_serie(serie):
    """Calculates the annual compounded return of a price series.

    The function takes a price series and calculates the annual compounded
    percentage return. The series is expected as an object that has a
    chronologically ordered index.

    Args:
        serie (pandas.Series): A chronologically ordered price series.

    Returns:
        float: The annual compounded percentage return of the series.

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
    """Calculates the annual return of a price series for different periods.

    The function takes a price series and calculates the annual compounded
    percentage return for the total period, last 3 years, last 5 years, and last
    10 years.

    Args:
        serie (pandas.Series): A chronologically ordered price series.

    Returns:
        tuple: A tuple containing the annual compounded percentage returns for the
        total period, last 3 years, last 5 years, and last 10 years, respectively.

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
    """Generates an HTML report with corrected CAGR and annual return information.

    The function uses the `qs.reports.html` library to generate an HTML report
    comparing the portfolio return with a benchmark. After generating the HTML,
    it corrects the CAGR information in the generated HTML with correct annual
    returns for different periods (3 years, 5 years, 10 years, and all-time).

    Args:
        retorno_carteira (pandas.Series): Portfolio return series.
        benchmark (pandas.Series): Benchmark return series.
        benchmark_titulo (str): Title/name of the benchmark used.
        nome_arquivo (str): Name of the HTML file to be generated.
        titulo (str): Title of the report.

    Returns:
        None

    Example:
        >>> portfolio_returns = pd.Series([0.02, 0.03, -0.01, 0.02, 0.01])
        >>> benchmark_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.008])
        >>> Func_relatorio_html(portfolio_returns, benchmark_returns, "S&P 500", "relatorio.html", "Performance Report")
        HTML report generated successfully!
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
    return None
