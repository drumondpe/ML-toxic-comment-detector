# Desintoxifier

Este projeto se propõe a verificar até onde podemos ir na desintoxicação de comentários online, utilizando técnicas de aprendizado de máquina e processamento de linguagem natural. A ideia é transformar comentários tóxicos em versões mais neutras ou construtivas, mantendo o sentido original.
- Link para o vídeo de apresentação: [YouTube](https://youtu.be/twnb5KY1bRM)

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

## 🏁 Exemplo de Uso

Na pasta notebook, você encontrará um arquivo desintoxifier.ipynb que demonstra o uso do pipeline completo, desde a geração de embeddings até a desintoxicação e reconstrução do texto.


## 🔗 Referências

- [SONAR - Meta AI](https://github.com/facebookresearch/SONAR)
- [Cleaned Toxic Comments - Kaggle](https://www.kaggle.com/datasets/fizzbuzz/cleaned-toxic-comments)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Fairseq2 - Meta AI](https://facebookresearch.github.io/fairseq2/stable/)
