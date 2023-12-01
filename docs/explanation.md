
## Descrição do Projeto

O projeto abrange uma série de etapas fundamentais para análise e montagem de carteira de investimentos, priorizando o uso de dados fundamentais das empresas. A seguir, estão resumidas as principais fases do projeto:

### 1. Coleta de Dados

- **Origem dos Dados:** O projeto inicia com a coleta de dados das ações a partir do banco de dados da plataforma "comdinheiro". Esses dados são essenciais para análise fundamentalista das empresas.

### 2. Tratamento de Dados

- **Dados do Balanço Anualizado e Normalizado:** Cada ação é submetida a um processo de tratamento, proporcionando informações cruciais sobre o balanço das empresas. Isso inclui métricas anualizadas e normalizadas por ação, como Lucro por Ação (LPA) e EBITDA por Ação.

- **Múltiplos das Empresas:** Um segundo DataFrame é gerado, fornecendo os múltiplos das empresas. Esses múltiplos incluem indicadores como Preço sobre Lucro (P/L), EV/EBITDA e Margem Líquida, proporcionando uma visão abrangente da saúde financeira das empresas.

- **Crescimento dos Resultados (CAGR):** Calcula-se o crescimento anual composto (CAGR) para métricas-chave, como Lucro, EBITDA e Receita, permitindo uma avaliação do desempenho histórico das empresas.

### 3. Avaliação e Sugestão de Carteira

- **Retorno Esperado de Ativos:** Utilizando uma abordagem abrangente, o projeto calcula o retorno esperado de cada ativo. Isso leva em consideração múltiplos, dívida, taxa de crescimento, dividendos e a trajetória da taxa SELIC.

- **Otimização da Carteira:** Com base nos retornos esperados, é sugerida uma alocação de ativos para maximizar o índice de Sharpe, proporcionando uma carteira otimizada para o dia em questão.

### 4. Backtesting e Análise

- **Backtesting da Estratégia:** A estratégia proposta é testada retrospectivamente para avaliar seu desempenho ao longo do tempo. Isso inclui a alocação de ativos ao longo do período de backtesting.

- **Análise via Planilha Excel:** Uma planilha Excel é gerada para uma análise detalhada da alocação ao longo do período de backtesting, fornecendo insights valiosos sobre o desempenho da estratégia.

Este projeto integra uma abordagem abrangente, desde a coleta de dados até a avaliação da estratégia proposta, proporcionando uma base sólida para tomada de decisões de investimento.


## Justificativa do Projeto

Este projeto foi concebido com o objetivo primordial de desenvolver uma estratégia de investimento robusta, centrada principalmente em dados fundamentalistas, visando a construção de uma carteira alinhada à tendência da curva de juros brasileira. A motivação por trás desse empreendimento pode ser resumida em alguns pontos-chave:

### 1. **Aproveitar as Oscilações da Curva de Juros**

A curva de juros brasileira é notável por suas oscilações significativas e de longo prazo. Ao utilizar dados fundamentalistas, busca-se identificar oportunidades estratégicas que podem ser influenciadas por mudanças nessa curva. A estratégia busca otimizar a alocação de ativos em diferentes ambientes de taxa de juros.

### 2. **Identificação de Ativos de Valor Sustentável**

A ênfase na análise fundamentalista visa identificar ativos que demonstram uma excelente capacidade de geração de valor ao longo dos anos. O mercado brasileiro oferece ações de empresas que conseguem entregar retornos sobre o capital investido superiores ao custo de capital, indicando um crescimento sustentável.

### 3. **Geração de Caixa e Estabilidade**

Ativos que apresentam forte geração de caixa e retorno sobre o capital investido são priorizados. Esses ativos tendem a ser menos voláteis e têm o potencial de proporcionar retornos mais elevados em comparação com o benchmark do Ibovespa. A estabilidade financeira dessas empresas é um fator essencial na construção de uma carteira resiliente.

### 4. **Maximização do Retorno sobre o Capital Investido**

O projeto procura otimizar o retorno sobre o capital investido, considerando não apenas os aspectos financeiros das empresas, mas também variáveis econômicas, como a trajetória da taxa SELIC. Isso contribui para a identificação de ativos que podem oferecer um retorno mais atrativo.

### 5. **Crescimento Sustentável ao Longo do Tempo**

Ao investir em ativos com crescimento sustentável ao longo do tempo, o projeto visa criar uma carteira equilibrada e resistente às flutuações do mercado. A análise de métricas fundamentais, como Lucro por Ação (LPA) e EBITDA por Ação, auxilia na seleção de ativos com histórico consistente de desempenho.


### 6. **Penalizações para Ativos com Alta Alavancagem**

A presença de alta dívida pode impactar negativamente o desempenho de um ativo e aumentar sua vulnerabilidade a condições adversas de mercado. Nesse sentido, o projeto aplica penalizações para ativos com elevada alavancagem, considerando que altos níveis de endividamento podem comprometer a capacidade da empresa de gerar caixa e enfrentar desafios financeiros.


### Conclusão

Em síntese, o projeto busca criar uma estratégia que vai além da simples especulação de curto prazo, concentrando-se na seleção criteriosa de ativos com base em fundamentos sólidos. A expectativa é que essa abordagem proporcione uma carteira bem equilibrada, capaz de enfrentar diferentes cenários do mercado e, ao mesmo tempo, buscar retornos superiores ao benchmark.











