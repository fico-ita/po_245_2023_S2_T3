# Bem-vindo ao FICO

Leia a documentação do Mkdocs em [mkdocs.org](https://www.mkdocs.org).

## Comandos

* `mkdocs new [dir-name]` - Cria um novo projeto.
* `mkdocs serve` - Inicia o servidor live-reloading de documentação.
* `mkdocs build` - Constrói o site da documentação.
* `mkdocs -h` - Imprime a mensagem de ajuda.

## Estrutura da documentação

    mkdocs.yml    # Arquivo de configuração.
    docs/
        index.md  # Página principal da documentação
        ...       # Outras páginas via markdown, imagens e outros arquivos.

## Organização da documentação

Para a organização da documentação de usuário, baseie-se na proposta de
[Diátaxis](https://diataxis.fr/), que consiste em 3 classes de documentos:


### Tutorials

A seção de tutoriais fornece orientações detalhadas sobre como utilizar o pacote FICO para análise e montagem de portfólio.

1. Download e Tratamento dos Dados:

   Este tutorial abrange o processo de obtenção e preparação dos dados fundamentais das ações necessários para a estratégia de montagem de portfólio. Saiba como baixar os dados da plataforma "comdinheiro" e realizar o tratamento necessário.

2. Backtesting da Estratégia:

   O segundo tutorial guia você através do processo de backtesting da estratégia de montagem de portfólio. Descubra como executar o backtesting e analisar os resultados para otimizar o portfólio.



---



### Reference

A seção de referência fornece detalhes sobre todas as funções disponíveis no pacote FICO. Consulte esta seção para obter informações técnicas sobre a implementação e os parâmetros de cada função.

---

### Explanation

Na seção de explicação, apresentamos o raciocínio por trás da construção da estratégia **FundAlphaPortfolioOptimizer**. Descubra as principais decisões tomadas durante o desenvolvimento, incluindo considerações sobre dados fundamentais, cálculos de retorno esperado, penalizações e a lógica por trás da otimização do portfólio para maximizar o índice de Sharpe. Este é um recurso valioso para entender a abordagem adotada e as premissas subjacentes à estratégia.

---


## FundAlphaPortfolioOptimizer

::: fico


## Projeto

A proposta do projeto aborda a fundamentação teórica e os motivos que levaram à concepção deste projeto, respaldada por relatos históricos. Além disso, apresenta a elaboração de uma fórmula que contribui para o cálculo do retorno esperado de um ativo. O escopo do projeto foi delineado para abranger as ações contidas no índice IBOVESPA. A metodologia adotada na proposta de projeto busca estimar o retorno esperado do ativo com um maior grau de discricionariedade, uma vez que incorpora variações significativas de acordo com o setor.

- [Proposta de Projeto](arquivos/PO245___Proposta_de_Projeto.pdf)

O projeto final passou por algumas modificações, incluindo uma metodologia de cálculo do retorno esperado estabelecida de maneira menos dependente de discricionariedade. Os resultados, o tratamento de dados e os fundamentos estão devidamente esclarecidos na versão final.

- [Projeto](arquivos/PO245___Versão_Final.pdf)


## Agradecimento

Quero expressar minha sincera gratidão aos Professores Elton Sbruzzi, Vitor Curtis e Michel Carlo Rodrigues Leles, cujo conhecimento, orientação e apoio foram fundamentais para o desenvolvimento deste projeto. Seu comprometimento com a educação e dedicação aos alunos são fontes de inspiração.

Também gostaria de estender meus agradecimentos ao monitor Renan Lima por seu valioso auxílio, paciência e contribuições que enriqueceram significativamente o processo de aprendizado.

Agradeço especialmente à comdinheiro pela generosa disponibilização de dados úteis, que foram essenciais para o desenvolvimento e sucesso deste projeto.

Obrigado por serem uma parte essencial do meu percurso acadêmico e por tornarem possível a realização deste projeto.

Com gratidão,

Paulo Fernando Guimarães Tupinambá


## Como citar

Similar ao README.md
