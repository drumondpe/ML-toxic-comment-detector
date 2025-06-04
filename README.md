# Desintoxifier 🧼🧠

Este projeto propõe uma solução inteligente para transformar automaticamente comentários tóxicos em versões neutras, mantendo o significado original. A abordagem combina embeddings semânticos e manipulação vetorial com geração de texto via modelo SONAR da Meta AI.

## 🔍 Problema

A toxicidade em plataformas online (insultos, discurso de ódio, etc.) afeta negativamente as interações. Nosso objetivo é oferecer um método que não apenas detecte, mas também **reescreva comentários tóxicos** de maneira ética e semântica.

## 📊 Dados

Utiliza o dataset [Cleaned Toxic Comments (Kaggle)](https://www.kaggle.com/datasets/fizzbuzz/cleaned-toxic-comments), contendo ~235 mil comentários com marcações binárias para:

- `toxic`, `severe_toxic`, `obscene`, `insult`, `threat`, `identity_hate`

## 🧠 Metodologia

1. **Geração de Embeddings**: usando `TextToEmbeddingModelPipeline` do SONAR.
2. **Classificação Multirrótulo**: prevê as categorias tóxicas para cada comentário.
3. **Desintoxicação Vetorial**: aplica vetores de correção com base em pesos das categorias tóxicas.
4. **Reconstrução**: converte o vetor resultante de volta para texto com `EmbeddingToTextModelPipeline`.

## 🏁 Execução

```bash
python main.py
```

Você será solicitado a digitar o índice do comentário que deseja desintoxicar.

## 📁 Estrutura do Projeto

```
desintoxifier_project/
├── main.py             # Pipeline principal
├── embeddings.py       # Geração de embeddings com SONAR
├── classifier.py       # Cálculo de vetores médios de toxicidade
├── detoxifier.py       # Aplica vetores de correção
├── decoder.py          # Reconstrução textual a partir de embedding
└── README.md           # Este arquivo
```

## 📌 Possíveis Expansões

- Avaliação com métricas BLEU, ROUGE, METEOR
- Suporte multilíngue (SONAR é multilingue!)
- Classificadores mais robustos (ex: Transformers supervisionados)
- Aplicação com dados reais de Twitter, Reddit, etc.

## 🔗 Referências

- [SONAR - Meta AI](https://github.com/facebookresearch/SONAR)
- [Cleaned Toxic Comments - Kaggle](https://www.kaggle.com/datasets/fizzbuzz/cleaned-toxic-comments)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Fairseq2 - Meta AI](https://facebookresearch.github.io/fairseq2/stable/)
