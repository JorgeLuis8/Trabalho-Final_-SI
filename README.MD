# Classificação de Câncer de Mama com Redes Neurais (MLP)

Este projeto implementa um classificador de câncer de mama utilizando Redes Neurais Artificiais (MLP - Multi-Layer Perceptron) com dados do repositório UCI Machine Learning.

## Sobre o Projeto

O objetivo é prever se um paciente possui câncer de mama (classificação binária) com base em características clínicas, utilizando técnicas de Machine Learning e avaliando o modelo com métricas robustas.

### Principais Características

- **Dataset**: Wisconsin Breast Cancer Database (UCI ML Repository)
- **Modelo**: MLP Classifier com 2 camadas ocultas (100 e 50 neurônios)
- **Técnicas Aplicadas**:
  - Hold-out (80% treino / 20% teste)
  - Normalização com StandardScaler
  - Validação estratificada

## Tecnologias Utilizadas
```
python >= 3.8
pandas
numpy
matplotlib
scikit-learn
ucimlrepo
```


O script irá:
1. Baixar automaticamente o dataset do UCI ML Repository
2. Dividir os dados em treino e teste (Hold-out)
3. Normalizar as features
4. Treinar o modelo MLP
5. Exibir métricas de avaliação
6. Gerar a matriz de confusão

## Resultados Esperados

O modelo avalia o desempenho através de:

- **Acurácia (Accuracy)**: Percentual de acertos totais
- **Precisão (Precision)**: Proporção de diagnósticos positivos corretos
- **Revocação (Recall)**: Capacidade de identificar casos reais de câncer
- **F1-Score**: Média harmônica entre precisão e revocação
- **Matriz de Confusão**: Visualização dos erros e acertos

### Saída Visual

O script gera uma matriz de confusão mostrando verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.

## Arquitetura do Modelo
```
Camada de Entrada → [100 neurônios] → [50 neurônios] → Saída (2 classes)
                     ↓ tanh          ↓ tanh           ↓ softmax
```

**Hiperparâmetros**:
- Função de ativação: `tanh`
- Otimizador: `lbfgs`
- Regularização: `alpha=0.5`
- Épocas máximas: `5000`

## Metodologia

### Hold-out
Separação estratificada de 20% dos dados para teste, garantindo que o modelo seja avaliado em dados nunca vistos durante o treinamento.

### Normalização
Aplicação do `StandardScaler` para padronizar features (média 0, desvio padrão 1), essencial para convergência de redes neurais.

### Validação
Métricas calculadas exclusivamente no conjunto de teste para avaliar a generalização do modelo.

## Estrutura do Código
```
cancer_classification.py
│
├── Carregamento de Dados (UCI ML Repo)
├── Hold-out Split (80/20)
├── Normalização (StandardScaler)
├── Treinamento do MLP
├── Avaliação (Métricas + Matriz de Confusão)
└── Visualização dos Resultados
```

## Interpretação das Métricas

- **Alta Precisão**: Poucos falsos positivos (não diagnosticar câncer incorretamente)
- **Alto Recall**: Poucos falsos negativos (não deixar casos de câncer sem diagnóstico)
- **F1-Score**: Equilíbrio entre precisão e recall

**Nota**: Em problemas médicos, o Recall é crítico para evitar que casos de câncer passem despercebidos.

## Contribuindo

Contribuições são bem-vindas. Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novos modelos
- Melhorar a documentação



## Referências

- [UCI ML Repository - Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/451)
- [Scikit-learn MLPClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)