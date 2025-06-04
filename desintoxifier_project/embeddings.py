import pandas as pd
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

def carregar_dados(path="data/train_preprocessed.csv", limite=150000):
    df = pd.read_csv(path)
    return df.iloc[:limite].copy()

def gerar_embeddings(df):
    model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    with torch.inference_mode():
        embeddings = model.predict(df['comment_text'].values, source_lang="eng_Latn")
    return embeddings
