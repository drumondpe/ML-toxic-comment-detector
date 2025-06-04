from embeddings import carregar_dados, gerar_embeddings
from classifier import classificar_toxicidade
from detoxifier import desintoxicar
from decoder import reconstruir_texto

if __name__ == "__main__":
    idx = int(input("Digite o índice do comentário para desintoxicar: "))

    df = carregar_dados()
    embeddings = gerar_embeddings(df)
    texto_original = df.iloc[idx]['comment_text']
    print("\nTexto original:", texto_original)

    _, vetores_categoria = classificar_toxicidade(df, embeddings)

    # Pesos dummy: assume peso 1.0 para todas as categorias presentes
    categorias = ['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']
    linha = df.iloc[idx]
    pesos = {cat: float(linha[cat]) for cat in categorias if linha[cat] == 1.0}

    texto_desejado = desintoxicar(embeddings[idx], pesos, vetores_categoria)
    texto_final = reconstruir_texto([texto_desejado])[0]

    print("\nTexto desintoxicado:", texto_final)
