import torch
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

def reconstruir_texto(embeddings):
    decoder = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    with torch.inference_mode():
        return decoder.predict(embeddings, target_lang="eng_Latn")
