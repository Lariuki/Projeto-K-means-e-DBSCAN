# Clusterização: InvestSP Prospect Advisor

Todo bom empreendedor deve entender que para fechar bons negócios é necessário segmentar o perfil mais assertivo de clientes, antes de iniciar a prospecção.

Diante disso, vamos apresentar uma solução de clusterização, que apoiará as empresas como consultorias, fornecedores de equipamentos, hardwares e softwares, entre outros. A formular a estratégia de prospects dentro do estado de São Paulo, a partir de parâmetros pré-estabelecidos como sub-região/município de atuação, tipo de investimento e ramo de atividade da empresa investidora, por meio de algoritmos em Python e integração de API.

Essa ferramenta irá servir de apoio às áreas de vendas e desenvolvimento de novos clientes das empresas usuárias da solução, principalmente para tomadores de decisão das áreas comerciais.

## O que é Clusterização?

> Clusterização é o agrupamento automático de instâncias por similaridade, uma classificação não-supervisionada dos dados. Ou seja, um algoritmo que clusteriza dados, classifica eles em conjuntos de dados que se ‘assemelham’ de alguma forma.

Para o processo de clusterização, existem 3 tipos de algoritmos não supervisionado mais populares que realizam essa operação:

1. **K-Means** - O algoritmo encontra **k clusters** diferentes no conjunto de dados, através de cálculos vetoriais, cada center point busca o centróide através da média das distâncias de todos os pontos do grupo. Ou seja, avalia e clusteriza os dados de acordo com suas características.

![](https://miro.medium.com/v2/resize:fit:960/format:webp/1*-1qqyoXnxHpKrkbWSk30sQ.gif)

2. **Mean-Shift**: O número de clusters é determinado pelo algoritmo em relação aos dados, diferente do K-Means. O algoritmo visa descobrir “bolhas” em uma densidade uniforme de amostras, baseado em centróides que atualiza clusters para serem a média dos pontos dentro de uma determinada região.

![](https://miro.medium.com/v2/resize:fit:1280/format:webp/1*4HWMFPLt0tNiSUgC3pl9Eg.png)

3. **(DBSCAN) Density-Based Spatial Clustering of Application with Noise**: Encontra amostras de núcleo de alta densidade e expande os clusters a partir delas. Ele também marca como outliers os pontos que estão em regiões de baixa densidade. Geralmente utilizados para localizar padrões e prever tendências.

![](https://miro.medium.com/v2/resize:fit:1350/format:webp/1*NbREY_FV2i-5U_ypQhNSRQ.gif)

## Metologia, o que vamos utilizar?

Como metodologia, utilizaremos o **Python** para a linguagem de programação e algumas de suas bibliotecas, técnicas variadas de ciências de dados e aprendizagem da máquina, como:

* [Análise exploratória de dados (EDA);](https://medium.com/@isnardgurgel/30d404f6244)
* Tratamento de dados (data cleansing);
* Análises gráficas;
* Algoritmos de Clusterização;
* Coeficiente de silhueta e Método de cotovelo;
* Análise business;

Ao lidar com o problema de agrupamento, iremos utilizar algoritmos para descobrir grupos significativos em nossos dados, a fim de oferecer um Roadmap dos potenciais prospects para usuários dessa solução.

Como ferramenta utilizaremos para fazer a codificação do algoritmo o Google Colaboraty (sinta-se livre para utilizar tanto o Colaboraty quanto o Jupyter da suíte Anaconda).

Para prover uma visão de negócios e alinhamento do problema/solução, segue abaixo data science canvas baseado no business canvas.

![](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*KO_8y5Dhtq_8DmjvlF4bTA.png)

## Dados, qual fonte vamos utilizar?

Como fonte de dados, utilizaremos a pesquisa **PIESP (Pesquisa de Investimento Anunciados no Estado de São Paulo)** divulgada pela Fundação SEADE, que capta diariamente anúncios de investimentos feitos na imprensa por empresas privadas e públicas.

Desde 2012, quando foi implantada a revisão metodológica da pesquisa, passaram a ser considerados todos os investimentos confirmados, independente de informações sobre valor de investimento, pressupondo-se que aqueles sem informações de valor também sinalizam a evolução da atividade econômica no Estado de São Paulo.

## Análise Exploratória de Dados

A EDA como é conhecida a Análise Exploratória de Dados, vai nos fazer entender melhor sobre nosso modelo de negócio ou mercado, dando um conhecimento sobre quais features são mais relevantes, o que precisamos acrescentar ou até mesmo excluir, identificar padrões, outliers, correlações.

Neste caso em especifico, notamos que precisaríamos imputar novos dados de “Setor” de cada empresa alvo de investimento.

Pesquisa de Investimentos Anunciados no Estado de São Paulo (Piesp): https://www.piesp.seade.gov.br/

### Tratamento de dados (data cleaning)

Para o tratamento de dados, criamos um novo dataset, segmentando somente as linhas com o valor de investimento e multiplicamos a coluna “Real (em milhões)” por 10000, para padronizarmos os dados. Basicamente isso poderia ser resumido pela utilização do **Feature Engineering**, que nada mais é que apenas a limpeza de dados ou o data cleaning.

### Análises gráficas

Em seguida, fizemos um group by e análises gráficas para nos aprofundar ainda mais.

> Para mais detalhes sobre a EDA, leia o artigo completo: [clicando aqui.](https://larissaakemi.medium.com/an%C3%A1lise-explorat%C3%B3ria-de-dados-eda-71b1bb4bc6ad)

## Algoritmos de classificação

Logo após a EDA, vamos partir para a etapa dos algoritmos de classificação, onde iremos trabalhar de forma mais aprofundada nas técnicas de machine learning não supervisionada, usada para dividir um grupo em conjuntos.

Neste artigo, iremos trabalhar dois tipos de algoritmos de classificação, que são: K-means e DBSCAN. Outra metodologia que testamos foi o Clustering Hierárquico.

### K-means vs DBSCAN

O K-means em sua essência, tenta organizar os dados em um número especificando clusters, o objetivo dele é identificar **pontos de dados semelhantes** e agrupa-los enquanto tenta distanciar cada agrupamento o máximo possível, mas é considerado extremamente vulnerável a outliers. Diferente do DBSCAN, o K-means exige que você especifique o número de clusters que deseja encontrar.

Já no DBSCAN (agrupamento espacial baseado em densidade de aplicativos com ruído), ele **agrupa pontos próximos** uns dos outros com base em uma medição de distância (distância euclidiana) e um número mínimo de pontos. Como seu nome já diz, o DBSCAN é campeão em lidar com outliers/ruídos, mas ele sofre com dados dimensionais elevados e exige que você especifique os parâmetros épsilon e pontos mínimos.

### Testando K-means e DBSCAN

Bom, para que o artigo não fique extenso, decidimos escrever um outro artigo sobre os testes realizados para os dois tipos de algoritmos para este dataset.

Contando um pouco como foi a nossa expectativa com os dois modelos, primeiramente aprendendo em uma das primeiras aulas da Tera sobre aprendizado da máquina, botamos muita expectativa no K-means. Quando iniciamos os testes dos algoritmos, tivemos uma primeira impressão, de que o DBSCAN teria dado mais certo no agrupamento dos clusters e no comportamento diante aos outliers, mas estávamos totalmente errados.

>Para mais detalhes sobre os testes dos algoritmos, leia o artigo completo: [clicando aqui.](https://larissaakemi.medium.com/testando-k-means-e-dbscan-investsp-prospect-advisor-139c57c47f59)

## Análise de Negócios e Conclusões dos Clusters Gerados

Baseado no objetivo de apoiar empresas a formular estratégias de Prospects dentro do estado de São Paulo, refizemos o processo de clusterização para K-means e DBSCAN com features “Setor”, “Faixa de valor” e “Região”.

Desta forma, conseguirmos gerar “Personas” pensando em como podemos beneficiar empresas na busca por novos clientes.

Neste caso o K-means se mostrou mais eficiente, gerando 4 clusters bem definidos pelas dimensões Setor, Faixa de valor de investimento e Região. Analisando os 4 Clusters gerados pelo K_Means conseguimos ter as seguintes Personas:

* **Persona 1 — Cluster 0:** Empresas/Fornecedores com interesse em investimentos P e M para os Setores Infra, Outros e Serviços

* **Persona 2 — cluster 1:** Empresas/Fornecedores com interesse em investimentos G e GG para os Setores Infra, Outros e Serviços

* **Persona 3 — cluster 2:** Empresas/Fornecedores com interesse em investimentos P e M para os Setores Comércio e Indústria

* **Persona 4 — cluster 3:** Empresas/Fornecedores com interesse em investimentos G e GG para os Setores Comércio e Indústria

Por fim, geramos um deploy da aplicação desenvolvida focando nos usuários finais, ou seja, possibilitando que fornecedores e consultores de um dos 4 perfis, possam utilizar os resultados para prospectar novos clientes.

Utilizamos a API Streamlit para interface Web + Códigos no GitHub e Heroku como servidor de aplicação. Acesse a este link: https://clustersinvestsp.herokuapp.com/

> Para mais detalhes sobre a análise de negócios, leia o artigo completo: [clicando aqui.](https://augustocamargos.medium.com/an%C3%A1lise-de-neg%C3%B3cios-e-conclus%C3%B5es-dos-clusters-gerados-investsp-prospect-advisor-bc498dc0f085)

