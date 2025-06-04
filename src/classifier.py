import numpy as np

CATEGORIAS = ['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']

def classificar_toxicidade(df, embeddings):
    df['embedding'] = [e.detach().cpu().numpy() for e in embeddings]
    vetores_categoria = {}

    for cat in CATEGORIAS:
        subset = df[df[cat] == 1]
        if not subset.empty:
            vetores_categoria[cat] = np.mean(np.stack(subset['embedding'].values), axis=0)

    neutros = df[(df[CATEGORIAS].sum(axis=1) == 0)]
    vetores_categoria['neutro'] = np.mean(np.stack(neutros['embedding'].values), axis=0)

    return None, vetores_categoria
