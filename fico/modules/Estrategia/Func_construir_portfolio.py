# %% [markdown]
# # Bibliotecas

# %%
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

# %% [markdown]
# # Funções

# %% [markdown]
# ## Funções básicas

# %% [markdown]
# ### Carregar Dados


# %%
def load_data_main(data_simulacao, IBOV=True):
    ## Se eu estiver rodando no diretório do script
    if __name__ == "__main__":
        # Obtém o diretório atual antes de alterá-lo
        diretorio_anterior = os.getcwd()
        # Divide o diretório anterior em partes separadas por '\\'
        partes_do_diretorio = diretorio_anterior.split("\\")
        # Crie um novo diretório excluindo o último item
        novo_diretorio = os.path.join(
            partes_do_diretorio[0],
            "\\" + "\\".join(partes_do_diretorio[1:-2]),
        )
        # Agora, o novo diretório conterá o caminho até o diretório anterior sem o último item
        os.chdir(novo_diretorio)
        # Importar a Biblioteca
        from modules.Colher_tratar_dados.load_data import load_data

        # Carregando os dados
        [dict_df_acoes_m, df_benchmark_m, df_Expectativa_Selic_mensal_m] = load_data(
            data_simulacao,
            IBOV,
        )

        # Após a conclusão, você pode restaurar o diretório original, se desejar
        os.chdir(diretorio_anterior)

    ## Se eu estiver rodando no diretório na main
    else:
        # Importar a Biblioteca
        from modules.Colher_tratar_dados.load_data import load_data

        # Carregando os dados
        [dict_df_acoes_m, df_benchmark_m, df_Expectativa_Selic_mensal_m] = load_data(
            data_simulacao,
            IBOV,
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

# %% [markdown]
# ## Função de Plotar gráficos


# %%
def Func_Plotar_Portfolio(df_alocacao):
    """Esta função gera um gráfico de pizza para visualizar a composição de um portfólio com base em um DataFrame de alocação.

    Parâmetros:
    df_alocacao (DataFrame): Um DataFrame contendo a alocação de ativos em um portfólio. Deve ter colunas nomeadas 'leftover' e 'Total'.

    Retorna:
    None

    Exemplo:
    Func_Plotar_Portfolio(df_alocacao)
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


# %% [markdown]
# ## Funções Penalizações


# %%
def Func_ajuste_reta(x_fit, x_inf, x_sup, y_inf, y_sup):
    """Realiza um ajuste linear entre dois pontos e calcula o valor correspondente em x_fit.

    Parâmetros:
    x_fit (float): O valor de x para o qual se deseja calcular y_fit.
    x_inf (float): O valor de x do ponto inferior.
    x_sup (float): O valor de x do ponto superior.
    y_inf (float): O valor de y correspondente ao ponto inferior.
    y_sup (float): O valor de y correspondente ao ponto superior.

    Retorna:
    y_fit (float): O valor de y ajustado para x_fit.

    Esta função realiza um ajuste linear entre dois pontos (x_inf, y_inf) e (x_sup, y_sup) e calcula o valor correspondente de y para um dado valor de x (x_fit) dentro desse intervalo. Se x_fit for menor que x_inf, o valor retornado será y_inf. Se x_fit for maior que x_sup, o valor retornado será y_sup. Para valores de x_fit dentro do intervalo (x_inf, x_sup), a função calcula o valor de y utilizando a equação da reta ajustada entre os pontos inferiores e superiores.

    Exemplo de uso:
    y_fit = Func_ajuste_reta(2.5, 2, 4, 3, 7)
    # Retorna o valor de y ajustado para x_fit = 2.5 com base nos pontos (2, 3) e (4, 7).
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


# %%
def Func_fator_juros(ind_aumento_selic, limite_inferior, limite_superior, Setor):
    """Calcula o fator de penalização com base no aumento da taxa Selic para empresas de diferentes setores.

    Parâmetros:
    - ind_aumento_selic (float): Indicador do aumento da taxa Selic.
    - limite_inferior (float): Limite inferior para o indicador de aumento da taxa Selic.
    - limite_superior (float): Limite superior para o indicador de aumento da taxa Selic.
    - Setor (str): Setor da empresa.

    Retorna:
    - float: Fator de penalização calculado.

    Descrição:
    A função calcula o fator de penalização com base no aumento da taxa Selic para empresas de diferentes setores.
    O cálculo considera diferentes ajustes lineares com base nos limites definidos e no indicador de aumento da taxa Selic.
    O fator de penalização é retornado.

    Exemplo:
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


# %% [markdown]
# ## Dados_iniciais


# %%
def Dados_iniciais(df_acao, df_multiplos, df_CAGR, df_benchmark):
    """Coleta dados iniciais necessários para o cálculo de outras funções.

    Args:
        df_acao (pd.DataFrame): DataFrame contendo dados fundamentalistas da ação normalizados pelo número de ações ex-tesouraria.
        df_multiplos (pd.DataFrame): DataFrame contendo os múltiplos da empresa (dados diários).
        df_CAGR (pd.DataFrame): DataFrame contendo a taxa de crescimento de dados fundamentalistas.


    Returns:
        pd.Series: Série de dados iniciais coletados para uso em outras funções.

    A função coleta informações como setor, ROE, DY médio, ticker, CAGR médio, múltiplo atual, múltiplo médio,
    desvio do múltiplo médio e expansão do múltiplo. Esses dados são usados em outras funções para análise.

    Exemplo de uso:
    dados_iniciais = Dados_iniciais(df_acao, df_multiplos, df_CAGR, serie_estrategia_Setor)
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


# %% [markdown]
# ## Funções da estratégia

# %% [markdown]
# ### Retorno por HPR


# %%
def ret_HPR(df_acao, df_multiplos, df_CAGR, razao_Selic, df_benchmark):
    """Calcula a rentabilidade esperada de ativos baseada no crescimento do EBITDA, DY e expansão do múltiplo, com penalizações para baixo ROE e alta dívida líquida/EBITDA.

    Args:
        df_acao (pd.DataFrame): DataFrame contendo dados fundamentalistas da ação normalizados pelo número de ações ex-tesouraria.
        df_multiplos (pd.DataFrame): DataFrame contendo os múltiplos da empresa (dados diários).
        df_CAGR (pd.DataFrame): DataFrame contendo a taxa de crescimento de dados fundamentalistas.
        razao_Selic (float):
        df_benchmark (pd.DataFrame):

    Returns:
        pd.Series: Uma série contendo informações, incluindo setor, retorno anual esperado, HPR, ROE, dívida líquida/EBITDA, DY médio, CAGR médio, expansão do múltiplo, múltiplo atual, múltiplo médio, desvio do múltiplo e informações de retorno.
        pd.DataFrame: Um DataFrame com informações sobre o ativo.

    A função calcula o retorno esperado dos ativos com base no crescimento do EBITDA, DY e expansão do múltiplo, com penalizações para baixo ROE e alta dívida líquida/EBITDA.

    Exemplo de uso:
    dados_return = ret_HPR(df_acao, df_multiplos, df_CAGR, razao_Selic, estrategia_Setor)
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

    # Calculando CAGR de longo prazo e medio prazo em função do CAGR_ordenado
    if CAGR_ordenado == 1:
        CAGR_longo = CAGR_Medio_6
        CAGR_curto = CAGR_Medio_4 * 0.7 + CAGR_Medio_2 * 0.3
    elif CAGR_ordenado == 0:
        CAGR_longo = CAGR_Medio_8
        CAGR_curto = CAGR_Medio_4
    else:
        CAGR_longo = CAGR_Medio_8 * 0.7 + CAGR_Medio_6 * 0.3
        CAGR_curto = CAGR_Medio_4

    # Se Cíclica seja mais conservador
    ciclica = False
    if (ML_1std <= 0) or (M_EBITDA_1std <= 0.02):
        ciclica = True

    # Cálculo do  CAGR_util e DY_util
    if ciclica:
        ### Cálculo do CAGR_util
        CAGR_util = CAGR_longo * (0.5 * Ind_alta_MM) + CAGR_Medio_min * (
            1 - 0.5 * Ind_alta_MM
        )

    else:
        ### Cálculo do CAGR_util
        CAGR_util = (
            CAGR_longo * (0.3 + 0.3 * Ind_alta_MM)
            + CAGR_curto * (0.4 * Ind_alta_MM)
            + CAGR_Medio_min * (0.7 - 0.7 * Ind_alta_MM)
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


# %% [markdown]
# ### main retorno

# %% [markdown]
# #### Funções de Suporte para main retorno


# %%
def Func_info_main(df_retorno, df_cotacao_ajustado, df_benchmark):
    """Calcula a rentabilidade esperada de ativos com diferentes estratégias dependendo do setor.

    Parâmetros:
    dict_df_acoes (dict): Dicionário com chaves sendo os ativos e valores contendo (df_acao, df_multiplos, df_CAGR).
    estrategia_Setor (pd.DataFrame): DataFrame contendo estratégias por setor.
    df_Expectativa_Selic_mensal (pd.DataFrame): DataFrame com a expectativa da SELIC mensal.
    df_benchmark (pd.DataFrame): DataFrame contendo informações do benchmark.

    Retorna:
    info_main (list): Lista contendo informações para o processo principal, incluindo o DataFrame de retorno esperado, cotações ajustadas e setores.
    info_verificao (list): Lista contendo informações para verificação, incluindo o DataFrame de retorno, cotações ajustadas, figuras e lista de excluídos.

    A função calcula o retorno esperado dos ativos com diferentes estratégias dependendo do setor em que estão inseridos.
    Também fornece informações para verificação e informações essenciais para o processo principal.

    Exemplo de uso:
    info_main, info_verificao = main_ret(dict_df_acoes, estrategia_Setor, df_Expectativa_Selic_mensal, df_benchmark, 0.05)
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
        if tem_null is False:
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
    """Calcula a rentabilidade esperada de ativos com diferentes estratégias dependendo do setor.

    Parâmetros:
    dict_df_acoes (dict): Dicionário com chaves sendo os ativos e valores contendo (df_acao, df_multiplos, df_CAGR).
    estrategia_Setor (pd.DataFrame): DataFrame contendo estratégias por setor.
    razao_Selic (float): Float contendo a razão entre a expectativa da SELIC e o valor atual da previsão.
    df_benchmark (pd.DataFrame): DataFrame contendo informações do benchmark.
    corte_rent (float): Valor mínimo de retorno esperado para considerar um ativo no portfólio.

    Retorna:
    info_main (list): Lista contendo informações para o processo principal, incluindo o DataFrame de retorno esperado, cotações ajustadas e setores.
    info_verificao (list): Lista contendo informações para verificação, incluindo o DataFrame de retorno, cotações ajustadas, figuras e lista de excluídos.

    A função calcula o retorno esperado dos ativos com diferentes estratégias dependendo do setor em que estão inseridos.
    Também fornece informações para verificação e informações essenciais para o processo principal.

    Exemplo de uso:
    info_main, info_verificao = main_ret(dict_df_acoes, estrategia_Setor, df_Expectativa_Selic_mensal, df_benchmark, 0.05)
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
    """Coleta dados iniciais para a construção do portfólio.

    Parâmetros:
    info_main (list): Lista de informações essenciais, incluindo o DataFrame de retorno esperado, cotações ajustadas e setores.

    Retorna:
    expec_retorno (pd.Series): Série com os retornos esperados dos ativos.
    cotacoes (pd.DataFrame): DataFrame com as cotações ajustadas dos ativos.
    setores (pd.Series): Série com informações sobre o setor de cada ativo.
    taxa_livre_risco (float): Taxa livre de risco, geralmente associada à SELIC.
    latest_prices (pd.Series): Série com os preços mais recentes dos ativos.

    A função coleta informações iniciais necessárias para a construção do portfólio, incluindo retornos esperados, cotações ajustadas, informações sobre setores, taxa livre de risco e preços mais recentes.

    Exemplo de uso:
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
    """Monta as limitações de alocação do portfólio com base nos setores.

    Parâmetros:
    setores (pd.Series): Série contendo informações sobre o setor de cada ativo.
    maior_per_ativo (int, opcional): Percentagem máxima de alocação em cada ativo. Padrão é 5%.
    maior_per_setor (int, opcional): Percentagem máxima de alocação em cada setor. Padrão é 20%.
    min_RF (float, opcional): Percentagem mínima de alocação em Renda Fixa. Padrão é 0.2 (20%).

    Retorna:
    setores_dic (dict): Dicionário com os setores e suas alocações correspondentes.
    sector_lower (dict): Dicionário com os valores mínimos de alocação por setor.
    sector_upper (dict): Dicionário com os valores máximos de alocação por setor.

    A função monta as limitações de alocação do portfólio com base nos setores dos ativos. Define a percentagem máxima de alocação permitida por ativo e por setor, bem como a percentagem mínima de alocação em Renda Fixa.

    Exemplo de uso:
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


# %%
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
    """Monta as limitações de alocação do portfólio com base nos setores.

    Parâmetros:
    setores (pd.Series): Série contendo informações sobre o setor de cada ativo.
    maior_per_ativo (int, opcional): Percentagem máxima de alocação em cada ativo. Padrão é 5%.
    maior_per_setor (int, opcional): Percentagem máxima de alocação em cada setor. Padrão é 20%.
    min_RF (float, opcional): Percentagem mínima de alocação em Renda Fixa. Padrão é 0.2 (20%).

    Retorna:
    setores_dic (dict): Dicionário com os setores e suas alocações correspondentes.
    sector_lower (dict): Dicionário com os valores mínimos de alocação por setor.
    sector_upper (dict): Dicionário com os valores máximos de alocação por setor.

    A função monta as limitações de alocação do portfólio com base nos setores dos ativos. Define a percentagem máxima de alocação permitida por ativo e por setor, bem como a percentagem mínima de alocação em Renda Fixa.

    Exemplo de uso:
    setores_dic, sector_lower, sector_upper = Func_alocacao_setorial(setores)
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


# %%
def Func_alocacao_dia(
    info_main,
    Montante_Total,
    maior_per_ativo=0.1,
    maior_per_setor=0.20,
    min_RF=0.2,
    gamma_tun=5,
    menor_per_aloc=0.005,
):
    """Calcula a alocação diária de ativos em um portfólio com base nas restrições de setor, risco e outras.

    Parâmetros:
    Capital (float): O montante total disponível para investimento.
    Pos_Short (float): A porcentagem de short que pode ser usada no portfólio.
    info_main (list): Lista contendo informações para o processo principal, incluindo retornos esperados, cotações ajustadas e setores.
    maior_per_ativo (float, opcional): A alocação máxima permitida por ativo em percentagem. Padrão é 5%.
    maior_per_setor (float, opcional): A alocação máxima permitida por setor em percentagem. Padrão é 20%.
    min_RF (float, opcional): A alocação mínima permitida em renda fixa em percentagem. Padrão é 0.2%.
    gamma_tun (float, opcional): Parâmetro de penalização da distribuição de pesos nulos. Padrão é 5.
    menor_per_aloc (float, opcional): Percentagem mínima de alocação para manter um ativo no portfólio. Padrão é 0.5%.

    Retorna:
    df_alloc_money (pd.DataFrame): DataFrame contendo a alocação de dinheiro em cada ativo em um determinado dia. As colunas representam os ativos e as linhas representam datas.

    A função calcula a alocação diária de ativos em um portfólio com base em várias restrições, incluindo alocação máxima por ativo, alocação máxima por setor e alocação mínima em renda fixa. Ela também permite a especificação de uma porcentagem de short no portfólio e aplica penalizações para evitar pesos nulos em ativos. O resultado é um DataFrame que mostra a alocação de dinheiro em cada ativo em datas específicas.

    Exemplo de uso:
    df_alloc_money, performance, weights_sem_nulos = Func_alocacao_dia(Capital, Pos_Short, info_main)
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
    """Constrói um portfólio de investimentos com base em diversas restrições, incluindo a expectativa de rentabilidade.

    Parâmetros:
    data_simulacao (str): A data para a simulação no formato 'YYYY-MM-DD'.
    corte_rent (float, opcional): A expectativa mínima de rentabilidade para um ativo entrar no portfólio. Padrão é 0.1.
    Capital (float, opcional): O montante total disponível para investimento.
    Pos_Short (float, opcional): A porcentagem de short que pode ser usada no portfólio. Padrão é 0.
    maior_per_ativo (float, opcional): A alocação máxima permitida por ativo em percentagem. Padrão é 5%.
    maior_per_setor (float, opcional): A alocação máxima permitida por setor em percentagem. Padrão é 20%.
    min_RF (float, opcional): A alocação mínima permitida em renda fixa em percentagem. Padrão é 0.2%.
    gamma_tun (float, opcional): Parâmetro de penalização da distribuição de pesos nulos. Padrão é 5.
    menor_per_aloc (float, opcional): Percentagem mínima de alocação para manter um ativo no portfólio. Padrão é 0.5%.

    Retorna:
    df_alloc_money (pd.DataFrame): DataFrame contendo a alocação de dinheiro em cada ativo em datas específicas.

    A função constrói um portfólio de investimentos com base em diversas restrições, incluindo a expectativa de rentabilidade. Ela carrega dados com a data de simulação, calcula a rentabilidade esperada dos ativos, aplica restrições com base em setores e outras métricas, e fornece a alocação de dinheiro em cada ativo em datas específicas.

    Exemplo de uso:
    df_alloc_money = Func_construir_porfolio(data_simulacao, corte_rent, Capital, Pos_Short, maior_per_ativo, maior_per_setor, min_RF, gamma_tun, menor_per_aloc)
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
