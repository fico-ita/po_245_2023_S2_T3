
## Tutorial: Backtesting e Geração de Relatório HTML

Nesta etapa, realizaremos o backtesting da estratégia e geraremos um relatório HTML para análise.

### Passo 1: Importar Módulos Necessários

```python
from fico.Back_test import Func_retorno_carteira
from fico.Backtesting_Module_generate_html import Func_relatorio_html
from datetime import datetime
```

Importe as funções `Func_retorno_carteira` e `Func_relatorio_html` necessárias para realizar o backtesting e gerar o relatório HTML.

### Passo 2: Configurar Parâmetros e Executar Backtesting

```python
# Definir parâmetros
benchmark_titulo = "IBOV"
nome_arquivo = "Estrategia_Fundamentalista.html"
titulo = "Portfolio Máximo Sharpe"

data_inicial = datetime(2013, 1, 1)
data_final = datetime(2023, 11, 3)

# Executar backtesting
retorno_carteira, serie_IBOV, _, _, _, _, _, _ = Func_retorno_carteira(
    data_simulacao_inicial=data_inicial,
    data_simulacao_final=data_final,
    Capital_Inicial=100000,
    dias_rebalancear=60,
    maior_per_ativo=0.5,
    maior_per_setor=0.5,
    min_RF=0.0,
    gamma_tun=1,
    menor_per_aloc=0.005,
)
```

Chame a função `Func_retorno_carteira` com os parâmetros desejados para realizar o backtesting. Os resultados serão armazenados nas variáveis `retorno_carteira`, `serie_IBOV` e outras.

### Passo 3: Gerar Relatório HTML

```python
# Gerar relatório HTML
Func_relatorio_html(
    retorno_carteira, serie_IBOV, benchmark_titulo, nome_arquivo, titulo
)
```

Chame a função `Func_relatorio_html` para gerar o relatório HTML com base nos resultados do backtesting.

### Observações Importantes

- Certifique-se de ter os módulos `Back_test` e `Backtesting_Module_generate_html` disponíveis no ambiente de execução.
- Ajuste os parâmetros conforme necessário para a sua estratégia.

Ao seguir esses passos, você terá o backtesting realizado e um relatório HTML gerado para análise.

---
**Nota:** Analise cuidadosamente os resultados do backtesting e use as informações geradas para otimizar sua estratégia.

Este tutorial é parte do processo de análise de dados no contexto do projeto.

---


## Tutorial: Exportação de Resultados para Excel

Nesta etapa, exportaremos os resultados do backtesting para um arquivo Excel para uma análise mais detalhada.

### Passo 1: Importar Módulo Necessário

```python
from fico.Backtesting_export_excel import Func_exportar_excel
```

Importe a função `Func_exportar_excel` necessária para exportar os resultados para um arquivo Excel.

### Passo 2: Exportar para Excel

```python
# Exportar para Excel
try:
    Func_exportar_excel(
        info_verificao_total,
        df_benchmark,
        df_exportar,
        indices_benchmark,
        excel_file="Alocação.xlsx",
    )
except:
    input("Fechar o Excel e apertar Enter para continuar")
    Func_exportar_excel(
        info_verificao_total,
        df_benchmark,
        df_exportar,
        indices_benchmark,
        excel_file="Alocação.xlsx",
    )
```

Chame a função `Func_exportar_excel` passando os resultados e informações necessários. Em caso de problemas ao exportar, a função solicitará que você feche o Excel antes de tentar novamente.

### Observação Importante

- Certifique-se de ter o módulo `Backtesting_export_excel` disponível no ambiente de execução.

Ao seguir esses passos, você terá os resultados do backtesting exportados para um arquivo Excel para uma análise mais detalhada.

---
**Nota:** Analise cuidadosamente os resultados exportados para extrair insights valiosos.

Este tutorial é parte do processo de análise de dados no contexto do projeto.

---
**Próximo Passo:** [Tutorial - Análise dos Resultados e Otimização da Estratégia](#) (A ser criado)
