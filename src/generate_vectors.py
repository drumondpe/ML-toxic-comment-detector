import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Carregar CSV original
df = pd.read_csv("../ML-toxic-comment-detector/data/train_preprocessed.csv")

# Carregar modelo
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Gerar embeddings em lote
tqdm.pandas()
embeddings = model.encode(df["comment_text"].tolist(), convert_to_numpy=True, show_progress_bar=True)

# Salvar os vetores como listas no CSV
df["embedding"] = [emb.tolist() for emb in embeddings]
df.to_csv("../ML-toxic-comment-detector/data/train_embedded.csv", index=False)