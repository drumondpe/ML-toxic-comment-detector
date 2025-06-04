import numpy as np

def desintoxicar(embedding, pesos, vetores_categoria):
    embedding_editado = embedding.detach().cpu().numpy().copy()
    for cat, peso in pesos.items():
        vetor_categoria = vetores_categoria.get(cat)
        if vetor_categoria is not None:
            embedding_editado += peso * (vetores_categoria['neutro'] - vetor_categoria)
    return embedding_editado
