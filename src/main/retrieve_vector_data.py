from database_connect_embeddings import TextEmbedding


def search_embeddings(query_embedding, session, limit=5):
    # 1. Skapa distans-uttrycket som en variabel
    distance_expr = TextEmbedding.embedding.cosine_distance(query_embedding)
    # 2. Skicka med uttrycket direkt i order_by istället för strängen 'distance'
    results = (
        session.query(
            TextEmbedding.id,
            TextEmbedding.sentence_number,
            TextEmbedding.content,
            TextEmbedding.file_name,
            distance_expr.label("distance"),
        )
        .order_by(distance_expr)
        .limit(limit)
        .all()
    )  # <-- Ändrat här
    return results
