# Classificador de Espécies de Peixes

Aplicação desenvolvida como trabalho da disciplina de Deep Learning do Mestrado em Engenharia Elétrica na Universidade do Estado do Amazonas (UEA).

## Contexto Acadêmico
- **Instituição:** Universidade do Estado do Amazonas (UEA)
- **Programa:** Mestrado em Engenharia Elétrica
- **Disciplina:** Deep Learning
- **Professor:** Dr. Tiago Melo
- **Alunos:** Chrystian Caldas, Edward Junior, Fabio Braz, Isabella Cabral, Rhedson Esashika

## Dataset Utilizado
- **Nome:** A Large Scale Fish Dataset  
- **Fonte:** Kaggle - https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset  
- **Descrição:** Conjunto com 9.000 imagens rotuladas de nove espécies de peixes. Para o experimento, os dados foram divididos em 70% treino (6.300), 15% validação (1.350) e 15% teste (1.350). Foi utilizado `SEED = 42` para garantir reprodutibilidade.

## Tabela Resumo Comparativa - Desempenho e Custo Computacional

| Modelo   | Arquitetura | Acurácia (Teste) | F1-Macro (Teste) | Balanced Acc. (Teste) | Épocas Treinadas | Tempo Total Treino (s) | Tempo Médio / Época (s) |
|----------|-------------|------------------|------------------|------------------------|------------------|------------------------|-------------------------|
| Modelo 1 | MobileNetV2 | 99.70%           | 0.9971           | 0.9972                 | 25               | 3091.32                | 123.65                  |
| Modelo 2 | InceptionV3 | 100.00%          | 1.0000           | 1.0000                 | 27               | 4883.23                | 180.86                  |

Salvando o resumo das métricas em `df_summary_results.csv`...  
Arquivo `df_summary_results.csv` salvo com sucesso.

## Relatório de Análise: Classificação de Peixes (MobileNetV2 vs. InceptionV3)

### 1. Introdução
Este experimento compara duas arquiteturas de CNN (MobileNetV2 e InceptionV3) usando transfer learning para a classificação de nove espécies de peixes. O objetivo foi avaliar a acurácia e o custo computacional de cada modelo.

### 2. Trabalhos Relacionados
Não documentado no notebook original; o foco esteve na implementação e avaliação experimental.

### 3. Metodologia
- **Dados:** 9.000 imagens do dataset A Large Scale Fish Dataset.
- **Divisão:** 70% treino (6.300), 15% validação (1.350) e 15% teste (1.350).
- **Reprodutibilidade:** `SEED = 42` aplicada a todas as divisões e embaralhamentos.
- **Modelos:** Bases pré-treinadas (MobileNetV2 e InceptionV3) congeladas; apenas o topo classificador foi treinado.
- **Hiperparâmetros:** Otimizador Adam (LR=0.001), batch size 32, loss `categorical_crossentropy`.
- **Callbacks:** `ModelCheckpoint (monitor='val_loss')`, `EarlyStopping (patience=10)` e `ReduceLROnPlateau (patience=3)`.

### 4. Resultados e Custo Computacional
Ambos os modelos atingiram desempenho elevado. O InceptionV3 alcançou 100% de acurácia no conjunto de teste, enquanto o MobileNetV2 chegou a 99.70%. Gráficos de curvas de aprendizado e matrizes de confusão foram gerados no notebook.

Tabela de comparação adicional:

| Modelo   | Arquitetura | Acurácia (Teste) | F1-Macro | Tempo Médio / Época (s) | Épocas Treinadas |
|----------|-------------|------------------|----------|--------------------------|------------------|
| Modelo 1 | MobileNetV2 | 99.70%           | 0.9971   | 124.74                   | 25               |
| Modelo 2 | InceptionV3 | 100.00%          | 1.0000   | 179.02                   | 38               |

### 5. Conclusão
O InceptionV3 atingiu métricas perfeitas, porém o MobileNetV2 foi cerca de 43.5% mais rápido por época e convergiu em menos épocas (25 contra 38), oferecendo um equilíbrio melhor entre custo computacional e desempenho quase perfeito.

#### 5.1 Limitações
Não detalhadas no notebook. Possíveis pontos: dataset relativamente simples ou ausência de fine-tuning completo.

#### 5.2 Próximos Passos
Não detalhados no notebook. Possibilidades: realizar fine-tuning do MobileNetV2 para buscar 100% de acurácia.

### 6. Apêndice: Hardware, Tempo de Treino e Configurações
- **Hardware:** Sessão Kaggle com GPU T4 dupla.
- **Memória GPU:** 2 x 15 GiB (total 30 GiB).
- **RAM da CPU:** 30 GiB.
- **Software:** Python (ambiente Kaggle), TensorFlow, Keras, Scikit-learn, Pandas.
- **Configurações de Treinamento:** `SEED=42`, `BATCH_SIZE=32`, `LEARNING_RATE=0.001` (com `ReduceLROnPlateau`), otimização com Adam, loss `categorical_crossentropy`.

**Tempo de Treino Detalhado**
- **Modelo 1 (MobileNetV2):** 25 épocas (parado por Early Stopping), 3118.51 s totais, 124.74 s por época.
- **Modelo 2 (InceptionV3):** 38 épocas (parado por Early Stopping), 6802.81 s totais, 179.02 s por época.
