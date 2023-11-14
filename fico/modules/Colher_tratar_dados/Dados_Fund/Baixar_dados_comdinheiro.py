# %% [markdown]
# ### Fazer o Download de todos os dados dos ativos elegíveis

# %%
## Bibliotecas
import json
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Lista de Ativos


# %%
def Baixar_dados(username, password, Fund, Cot, Eventos, lista_ativos):
    if lista_ativos == ["Todos"]:
        ## Ler os ativos que serão buscados
        Lista_ativos_busca = pd.read_parquet(
            "dataset/BR/ACOES/Dados_Brutos/Lista_Ativos_Busca.parquet",
        )
        Lista_ativos_busca = Lista_ativos_busca["Ticker"].to_list()
        Lista_ativos_busca.sort()

        Lista_ativos_busca2 = pd.read_parquet("dataset/BR/ACOES/IBrA.parquet")
        Lista_ativos_busca2 = Lista_ativos_busca2.index.to_list()
        Lista_ativos_busca2.sort()

        Lista_ativos_busca_Total = Lista_ativos_busca + Lista_ativos_busca2
        Lista_ativos_busca_Total.sort()
        # Converta a lista em um conjunto (set) para obter valores únicos
        Lista_ativos_busca_Total = set(Lista_ativos_busca_Total)

        # O conjunto (set) agora contém valores únicos

        # Se desejar, converta o conjunto de volta em uma lista
        Lista_ativos_busca_Total = list(Lista_ativos_busca_Total)
        Lista_ativos_busca_Total.remove("TIET11")
    else:
        Lista_ativos_busca_Total = lista_ativos

    #  Configuração da API
    url = "https://www.comdinheiro.com.br/Clientes/API/EndPoint001.php"

    querystring = {"code": "import_data"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    ## Modificar o nome das colunas para o padrão do banco de dados
    lista_colunas = [
        "Empresa",
        "Data_balanco",
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
        "BOLSA",
        "CODIGO_CVM",
        "Preco_fechamento",
        "Segmento",
        "Setor",
        "Setor_Comdinheiro",
        "Subsetor",
        "SubSubsetor",
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
        "Tipo_Balanco",
        "Convencao",
        "Moeda",
        "Data_demonstracao",
        "meses",
        "Data_analise",
    ]

    acumular = "3"
    moeda = "BRL"
    # Dados Fundamentalistas
    if Fund is True:
        for ticker in Lista_ativos_busca_Total:
            payload = (
                f"username={username}&password={password}&URL=HistoricoIndicadoresFundamentalistas001.php%3F%26"
                f"data_ini%3D31121994%26data_fim%3D20092024%26trailing%3D{acumular}%26conv%3DMIXED%26moeda%3D{moeda}%26c_c%"
                f"3Dconsolidado_preferencialmente%26m_m%3D1%26n_c%3D2%26f_v%3D1%26papel%3D{ticker}%26indic%3DNOME_EMPRESA%"
                f"2BDATA_ENTREGA_DEM_PRIM%2BTOTAL_ACOES_EX_TESOURARIA%2Bfator_equivalencia_acoes%2BFATOR_COTACAO%2BMARKET_VALUE%"
                f"2BPL%2BRL%2BEBITDA%2BDEPRE_AMOR%2BEBIT%2BLL%2BLL_SOCIO_CONTROL%2BLL_SOCIO_NAO_CONTROL%2BROIC%2BROE%2BDIVIDA_BRUTA%"
                f"2BDIVIDA_LIQUIDA%2BFINANCIAMENTO_ARREND_FIN%2BFCO%2BFCI%2BFCF%2BBOLSA%2BCODIGO_CVM%2BPRECO_FECHAMENTO%2BSEGMENTO%"
                f"2BSETOR%2BSETOR_COMDINHEIRO%2BSUBSETOR%2BSUBSUBSETOR%2BPAYOUT%2BPROVENTO%2BJCP%2BDY_12M%2BDY_24M%2BDY_36M%2BDY_48M%2BDY_60M%"
                f"2Bret_12m_aa%2Bret_01m_aa%2Bret_ano_atual%2Bret_cdi_01m%2Bret_cdi_12m%2Bret_cdi_ano_atual%2Bret_ibov_01m%2Bret_ibov_12m%"
                f"2Bret_ibov_ano_atual%26periodicidade%3Dtri%26graf_tab%3Dtabela%26desloc_data_analise%3D1%26flag_transpor%3D0%26c_d%3Dd%"
                f"26enviar_email%3D0%26enviar_email_log%3D0%26cabecalho_excel%3Dmodo1%26relat_alias_automatico%3Dcmd_alias_01&format=json2"
            )

            response = requests.request(
                "POST",
                url,
                data=payload,
                headers=headers,
                params=querystring,
            )

            try:
                # Acrescentar as seguintes linhas no final do script
                dict = json.loads(response.text)

                # Obs.: Sustituir ["chave"] pela chave correspondente
                df_ticker = pd.DataFrame(dict["resposta"]["tab-p0"]["linha"])
            except:
                raise

            nome_arquivo = f"dataset/BR/ACOES/Dados_Brutos/{ticker}_Fund.parquet"
            df_ticker = df_ticker.set_index("data")
            df_ticker.columns = lista_colunas
            df_ticker = df_ticker.applymap(
                lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x,
            )
            df_ticker.to_parquet(nome_arquivo, engine="fastparquet")

    # Dados de cotação
    if Cot is True:
        for ticker in Lista_ativos_busca_Total:
            payload_cotacao = f"username={username}&password={password}&URL=HistoricoCotacaoAcao001-{ticker}-03011994-15092024-1-1&format=json2"
            response = requests.request(
                "POST",
                url,
                data=payload_cotacao,
                headers=headers,
                params=querystring,
            )

            try:
                # Acrescentar as seguintes linhas no final do script
                dict = json.loads(response.text)

                # Obs.: Sustituir ["chave"] pela chave correspondente
                df_cotacao = pd.DataFrame(dict["resposta"]["tab-p0"]["linha"])
            except:
                raise

            nome_arquivo = f"dataset/BR/ACOES/Dados_Brutos/{ticker}_Cot.parquet"
            # Data como index
            df_cotacao = df_cotacao.set_index("data")
            # Removendo as listas vazias
            df_cotacao = df_cotacao.applymap(
                lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x,
            )

            # Substitua os valores "nd" por NaN
            df_cotacao = df_cotacao.replace("nd", np.nan)

            # Renomeando as colunas
            df_cotacao.columns = [
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
                "Tipo",
                "Quantidade_em_aluguel",
                "Vol_aluguel(MM_R$)",
            ]

            # Remova as linhas com valores NaN no fechamento histórico
            df_cotacao = df_cotacao.dropna(subset=["Fech_Historico"], axis=0)
            df_cotacao.to_parquet(nome_arquivo, engine="fastparquet")

    # Salvar eventos
    if Eventos is True:
        for ticker in Lista_ativos_busca_Total:
            payload_prov = f"username={username}&password={password}&URL=HistoricoProventos-{ticker}-03011994-15092024-ultimo_dia_com-DIV_e_JCP-8-1&format=json2"

            response = requests.request(
                "POST",
                url,
                data=payload_prov,
                headers=headers,
                params=querystring,
            )

            try:
                # Acrescentar as seguintes linhas no final do script
                dict = json.loads(response.text)

            except:
                raise

            try:
                Nao_existe_provento = (
                    isinstance(dict["resposta"]["tab-p0"], list)
                    and len(dict["resposta"]["tab-p0"]) == 0
                )
                if Nao_existe_provento is False:
                    if isinstance(dict["resposta"]["tab-p0"]["linha"], type(dict)):
                        df_prov = pd.DataFrame([dict["resposta"]["tab-p0"]["linha"]])
                    else:
                        df_prov = pd.DataFrame(dict["resposta"]["tab-p0"]["linha"])

                    ## Proventos
                    nome_arquivo = (
                        f"dataset/BR/ACOES/Dados_Brutos/{ticker}_Prov.parquet"
                    )
                    # Data como index
                    df_prov = df_prov.set_index("ultimo_dia_com")
                    # Removendo as listas vazias
                    df_prov = df_prov.applymap(
                        lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x,
                    )

                    # Substitua os valores "nd" por NaN
                    df_prov = df_prov.replace("nd", np.nan)

                    # Renomeando as colunas
                    df_prov.columns = [
                        "Tipo_do_Provento",
                        "Empresa",
                        "Tipo",
                        "Valor_do_Provento",
                        "Último_preco_com",
                        "Provento_por",
                    ]

                    # Remova as linhas com valores NaN no Valor_do_Provento
                    df_prov = df_prov.dropna(subset=["Valor_do_Provento"], axis=0)
                    df_prov.to_parquet(nome_arquivo, engine="fastparquet")
            except:
                raise

            try:
                Nao_existe_evento = (
                    isinstance(dict["resposta"]["tab-p1"], list)
                    and len(dict["resposta"]["tab-p1"]) == 0
                )
                if Nao_existe_evento is False:
                    if isinstance(dict["resposta"]["tab-p1"]["linha"], type(dict)):
                        df_evento = pd.DataFrame([dict["resposta"]["tab-p1"]["linha"]])
                    else:
                        df_evento = pd.DataFrame(dict["resposta"]["tab-p1"]["linha"])

                    ## Eventos
                    nome_arquivo = (
                        f"dataset/BR/ACOES/Dados_Brutos/{ticker}_Eventos.parquet"
                    )
                    # Data como index
                    df_evento = df_evento.set_index("data_evento")
                    # Removendo as listas vazias
                    df_evento = df_evento.applymap(
                        lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x,
                    )

                    # Substitua os valores "nd" por NaN
                    df_evento = df_evento.replace("nd", np.nan)

                    # Remova as linhas com valores NaN no fechamento histórico
                    df_evento = df_evento.dropna(axis=1, how="all")

                    # Renomeando as colunas
                    df_evento.columns = [
                        "Nome_Empresa",
                        "ClasseAcao",
                        "Evento",
                        "Fator",
                    ]
                    df_evento.to_parquet(nome_arquivo, engine="fastparquet")
            except:
                raise

            try:
                Nao_existe_subscricao = (
                    isinstance(dict["resposta"]["tab-p2"], list)
                    and len(dict["resposta"]["tab-p2"]) == 0
                )
                if Nao_existe_subscricao is False:
                    if isinstance(dict["resposta"]["tab-p2"]["linha"], type(dict)):
                        df_subscricao = pd.DataFrame(
                            [dict["resposta"]["tab-p2"]["linha"]],
                        )
                    else:
                        df_subscricao = pd.DataFrame(
                            dict["resposta"]["tab-p2"]["linha"],
                        )
                    ## Proventos
                    nome_arquivo = (
                        f"dataset/BR/ACOES/Dados_Brutos/{ticker}_Subscricao.parquet"
                    )
                    # Data como index
                    df_subscricao = df_subscricao.set_index("data_evento")
                    # Removendo as listas vazias
                    df_subscricao = df_subscricao.applymap(
                        lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x,
                    )

                    # Substitua os valores "nd" por NaN
                    df_subscricao = df_subscricao.replace("nd", np.nan)

                    # Remova as linhas com valores NaN no fechamento histórico
                    df_subscricao = df_subscricao.dropna(axis=1, how="all")

                    # Renomeando as colunas
                    df_subscricao.columns = [
                        "Nome_Empresa",
                        "ClasseAcao",
                        "Evento",
                        "Fator",
                        "Preco_Subscricao",
                    ]
                    df_subscricao.to_parquet(nome_arquivo, engine="fastparquet")
            except:
                raise
