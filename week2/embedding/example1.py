from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["AI helps humans make better decisions.", "Artificial intelligence improves human decision-making.", "The weather is sunny today."]

embeddings = model.encode(sentences, convert_to_tensor=True)
similarity = util.cos_sim(embeddings, embeddings)
print(similarity)