# AI Fellowship Data Repository

- [AI Fellowship Data Repository](#ai-fellowship-data-repository)
  - [📝 Descrição do problema](#-descrição-do-problema)
  - [💻 Stack](#-stack)
  - [💡 Estratégia](#-estratégia)
    - [🤖 LLM: reduzindo latência e custos](#-llm-reduzindo-latência-e-custos)
    - [🤯 Heurística](#-heurística)
      - [Pressupostos adotados](#pressupostos-adotados)
      - [Cache](#cache)
      - [Workflow](#workflow)
  - [👨🏻‍💻 Como usar](#-como-usar)
  - [🧩 Melhorias reconhecidas](#-melhorias-reconhecidas)


Esse repositório contém um projeto desenvolvido durante o processo seletivo para o fellowship promovido pela empresa [Enter](https://www.getenter.ai/).

## 📝 Descrição do problema

O desafio proposto gira em torno do problema de extração eficiente de pares chave-valor a partir de documentos desestruturados. Conforme apontado por [esse paper](https://arxiv.org/abs/2405.00505), por exemplo, *Key-Value Pair Extraction* é uma tarefa crítica cuja solução eficiente permanece em aberto.

## 💻 Stack

Por questões de familiaridade e agilidade no desenvolvimento/prototipação, optou-se pela linguagem Python.

Além disso, como modelo de linguagem (LLM), utilizou-se o [gpt-5-mini](https://platform.openai.com/docs/models/gpt-5-mini) da OpenAI.

## 💡 Estratégia

### 🤖 LLM: reduzindo latência e custos

**Nota**: no contexto de inferência de modelos de linguagem: `tokens consumidos = custo`. Portanto, um aumento/redução no número de tokens implica um aumento/redução proporcional no custo final.

Sabendo que a interação com um LLM seria uma peça fundamental e inegociável, o primeiro passo tomado durante o desenvolvimento foi testar formas de diminuir custo e latência (uma vez que chamadas a LLMs costumam ser o gargalo operacional e financeiro da operação):

1. Percebendo que a tarefa de identificar pares chave-valor não demanda uma linha de raciocínio muito elaborada, o primeiro teste feito foi retirar (ou, praticamente retirar) a feature de `reasoning` do modelo, setando `reasoning={"effort": "minimal"}` (os testes foram feitos passando o PDF como texto via prompt). 
    - Resultado: de ~20s para ~3s (**7x menos**) e de ~1600 tokens totais para ~400 (**4x menos**), sendo que o resultado permaneceu satisfatório.
    - Obs.: quando `effort` não é especificado, o valor padrão é `medium`.

2. Para evitar formatos de saída indesejados (o que geraria problemas desnecessários de JSON parsing), utilizou-se a feature de [Structured model outputs](https://platform.openai.com/docs/guides/structured-outputs), garantindo que o modelo sempre responderia conforme o modelo JSON estabelecido (utilizando a lib. `pydantic`).

3. Como uma tentativa de "enguxar" ainda mais o prompt, o esquema de entrada foi passado na estrutura YAML, que utiliza menos tokens que o formato JSON - o resultado não foi significativo, um vez que essa é um estrtégia crítica para cenários onde o JSON passado no prompt é extremamente longo, o que não é o caso médio do desafio.

4. Para aproveitar melhor o [Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) (reduzindo custo e latência), o prompt foi organizado de forma que as seções estáveis permaneçam no início, enquanto as partes variáveis são colocadas ao final, reduzindo a quantidade de conteúdo que precisa ser recarregado a cada requisição.

5. Por fim, testou-se passar o PDF de entrada de duas formas:
    1. Utilizando a feature de [File inputs](https://platform.openai.com/docs/guides/pdf-files?api-mode=responses) via base64, o que inevitavelmente aumenta custo e latência - uma vez que: "To help models understand PDF content, we put into the model's context both extracted text and an image of each page—regardless of whether the page includes images.", OpenAI.
    2. Utilizando apenas texto via engenharia de prompt. Realizar isso é complicado, uma vez que o layout desempenha um papel fundamental. Para contornar esse problema foi fornecido ao modelo um esquema que lhe permite entender o layout do arquivo original (aqui, começa a entrar a heurística utilizada, que será detalhada no próximo tópico) através de uma matriz. Exemplo para o arquivo `oab_1.pdf`:
   
            ```none
            Row 1: joana d'arc
            Row 2: inscrição | seccional | subseção
            Row 3: 101943 | pr | conselho seccional - paraná
            Row 4: suplementar
            Row 5: endereço profissional
            Row 6: avenida paulista, nº 2300 andar pilotis, bela vista
            Row 7: são paulo - sp
            Row 8: 01310300
            Row 9: telefone profissional
            Row 10: situação regular
            ```

        Apesar de modelos de linguagem serem, em essência, orientados a texto e não apresentarem desempenho ideal em dados tabulares, observou-se uma melhora significativa nos resultados quando as informações foram estruturadas em tabela/matriz, em comparação ao uso do texto corrido sozinho. Obviamente isso acabou resultando em um pequeno aumento de latência e tokens consumidos.
    
    **Resultados**: enviar o arquivo PDF para o LLM (via base64), em vez do texto extraído do PDF no prompt, resultou em aproximadamente **2x mais tempo**, **2x mais tokens**. Contudo, durante os experimentos, percebeu-se que os resultados foram um pouco inferiores e menos consistentes. Exemplos:
    - Para a chave `"situacao"` dentro de `"label": "carteira_oab"`: em alguns casos, o modelo retornou apenas `"regular"`, enquanto em outros retornou `"situação regular"`. Além disso, para a chave `"endereco_profissional"` dentro da mesma categoria: partes finais do endereço foram ocasionalmente omitidas — como, por exemplo, o CEP.

    O tópico a seguir apresenta a abordagem adotada para lidar com esses problemas.

### 🤯 Heurística

#### Pressupostos adotados

1. Conjunto definido de layouts por label.
    - Assumiu-se que documentos com mesma label tendem a possuir um conjunto de layouts padrão. Ou seja, para uma mesma label existe um conjunto de configurações a partir das quais os dados estão dispostos.
2. Mesma chave, mesmo tipo.
    - Assumiu-se valores de labels e chaves iguais possuem o mesmo tipo/formato.
    - Exemplo: dada uma label, uma chave `nome` sempre conterá uma string, uma chave `data` sempre conterá um valor no formato de data, uma chave `valor_total` sempre conterá um valor numério, etc..
3. LLM acerta.
    - Assume-se que o resultado gerado pela LLM quando alimentada com o arquivo PDF nativo está correto.

#### Cache

A cache é um dicionário cujos valores são preenchidos de forma adaptativa ao longo do processamento dos PDFs. Sua estrutura segue três níveis:

1. **Nível 1**: chaves correspondendo às *labels* dos documentos (ex.: `carteira_oab`, `tela_sistema`, etc.), permitindo que heurísticas sejam especializadas por tipo de documento.

2. **Nível 2**: cada label possui um dicionário como valor, cujas chaves correspondem às *keys* do esquema.

3. **Nível 3**: cada key possui um dicionário como valor, cujas chaves são:
    1. `count`, que armazena a quantidade total de vezes que a key foi solicitada em um esquema de requisição,
    2. `heuristics`, que corresponde a uma lista de heurísticas aprendidas (a ideia é que cada heurística seja útil para um layout específico),
    3. `type`, que corresponde ao tipo predominante do valor correspondente e
    4. `example_values`, que corresponde a uma lista de valores prévios.

4. **Nível 4**: cada heurística é um dicionário cujas chaves são:
    1. `position`: posição do valor na representação matricial do conteúdo do PDF (ver módulo `utils.pdf2mat.py`),
    2. `match_count`: número de vezes que essa heurística foi usada,
    3. Se o tipo for `string`, há também a chave `mean_length`: armazena um float com o tamanho médio acumulado dos valores da chave.

Exemplo da estrutura da cache:
```json
"carteira_oab": {
    "nome": {
        "count": 3,
        "heuristics": [
            {
                "position": [
                    0
                ],
                "match_count": 3,
                "mean_length": 11
            }
        ],
        "type": "string",
        "example_values": [
            "joana d'arc",
            "luis filipe araujo amaral",
            "son goku"
        ]
    },
    "inscricao": {
        "count": 3,
        "heuristics": [
            {
                "position": [
                    2,
                    0
                ],
                "match_count": 3
            }
        ],
        "type": "number",
        "example_values": [
            "101943"
        ]
    }
}
```

**Antes de realizar a chamada ao modelo** (gargalo do sistema em termos de custo e tempo) executa-se um pré-processamento por meio do método `heuristic_preprocessing()`. Esse método utiliza a cache de heurísticas já aprendidas para tentar preencher automaticamente parte do esquema de extração (`request_schema`) antes da inferência. Para cada chave do esquema, o método verifica se existem heurísticas previamente armazenadas para a label do documento atual e, se existir, tenta recuperar o valor correspondente consultando diretamente a matriz do PDF. Os valores recuperados são armazenados em um dicionário parcial (`partial_result`), que representa os campos resolvidos apenas por heurística, sem consulta ao modelo. Durante esse processo, o método também ajusta contadores internos e estatísticas de uso das heurísticas, reforçando aquelas que se mostram mais eficazes.

**Após a inferência do modelo**, o método `heuristic_update()` é responsável por atualizar a cache com os novos resultados obtidos. Ele registra o valor retornado, determina seu tipo, coleta exemplos representativos e identifica a posição do valor no PDF, transformando esse conhecimento em novas heurísticas. Se uma heurística existente já corresponder ao valor observado, sua frequência de acerto é incrementada; caso contrário, uma nova heurística é adicionada. O conjunto é então reordenado para priorizar heurísticas mais consistentes, mantendo apenas as mais relevantes para uso futuro.

Em resumo: 
- `heuristic_preprocessing()`: antecipa o que pode ser inferido sem o modelo
- `heuristic_update()`: permite que o sistema aprenda continuamente com novas extrações, tornando-o mais eficiente conforme mais documentos são processados.

#### Workflow

Os fluxogramas abaixo demonstram como os métodos `heuristic_preprocessing()` e `heuristic_update()`, respectivamente, funcionam.
 
```mermaid
flowchart LR
    A[Entrada: label, esquema de requisição e matriz do PDF] --> B
    B{Label presente na cache?}

    B -->|Não| C[Retorna dicionário vazio]
    B -->|Sim| D

    D[Para cada chave presente no esquema de requisição] --> E
    E{Há heurísticas para a chave?}
    E -->|Não| G[Nada é feito e muda para a próxima chave]
    E -->|Sim| F

    F[Para cada heurística presente na chave] --> H
    H{Acessa elemento na matriz do PDF com a posição armazenada pela heurística}

    H -->|Acesso inválido| I[Heurística não aplicável. Muda para a próxima heurística]
    H -->|Acesso válido| J

    J{Tipo armazenado pela herística compatível com o tipo do elemento acessado?}
    J -->|Não| K[Heurística não aplicável. Muda para a próxima heurística]
    J -->|Sim| L[Preenche par chave-valor no dicionário a ser retornado, onde chave = chave da requisição e valor = elemento acessado no PDF. Muda para a próxima chave]
```

```mermaid
flowchart LR
    A[Entrada: label, resultado da inferência e matriz do PDF] --> B{Label existe na cache?}

    B -->|Não| C[Criar entrada vazia para a label na cache]
    B -->|Sim| D

    C --> D

    D[Iterar sobre chave, valor do resultado] --> E{Valor vazio ou nulo?}
    E -->|Sim| F[Ignorar valor]
    E -->|Não| H

    H{Chave já existe na cache para esta label?}
    H -->|Não| I[Inicializar estrutura da chave]
    H -->|Sim| J

    I --> J

    J[Atualizar estatísticas da chave. Ex.: tipo predominante, contagem, exemplos representativos] --> K{Localizar posição do valor no PDF}

    K -->|Não encontrado| L[Encerrar atualização para esta chave]
    K -->|Encontrado| M{Existe heurística para esta posição e tipo?}

    M -->|Sim| N[Incrementar match_count e atualizar métricas]
    M -->|Não| O[Adicionar nova heurística]

    N --> P[Reordenar heurísticas por match_count]
    O --> P

    P[Manter apenas as N heurísticas mais fortes]
```

## 👨🏻‍💻 Como usar

1. Clone o repositório
```bash
https://github.com/joaoloss/ia-fellowship.git
cd ia-fellowship
```

2. 
```bash
uv init
```

2. 

## 🧩 Melhorias reconhecidas

Como o algoritmo é apenas um protótipo, é importante pontuar limitações/melhorias reconhecidas:

1. A versão atual da heurística construída **não identifica chaves ausentes**, o que aumenta a dependência do modelo de linguagem. Versões futuras poderiam contornar esse problema.
2. **Ausência de paralelismo/multithreading**: adicionar essa feature é um desafio que, infelizmente, não pôde ser solucionado por questão de prazo. Contudo, há alguns problemas que tornam a inserção dessa feature não trivial:
   1. Problema de sincronismo: ao processar múltiplos PDFs em paralelo, a ordem de processamento deixa de ser garantida, ou seja, a ordem de saída pode não corresponder à ordem de entrada.
   2. Efetividade reduzida da heurística: a heurística depende do acúmulo progressivo de informações — quanto mais documentos são processados, melhor ela fica. Entretanto, com múltiplas threads, documentos que são processados logo no início podem não se beneficiar da heurística simplesmente porque ela ainda não foi atualizada por outras threads. 
   
        Uma possível solução seria manter o processamento sequencial durante um determinado período ou até que um número mínimo de documentos tenha sido processado.