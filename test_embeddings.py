from sentence_transformers import SentenceTransformer
import numpy as np

# Embedding model load karna (pehli baar chalne pe download hoga)
print("Model load ho raha hai...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model load ho gaya!")

# Test sentences
sentences = [
    "He is a software engineer",
    "He works as a programmer",
    "The cat is sleeping on the mat"
]

# Sentences ko embeddings (numbers) mein convert karna
embeddings = model.encode(sentences)

print()
print("Embedding shape:", embeddings.shape)
print("Pehli 10 numbers (Sentence 1):", embeddings[0][:10])

# Cosine similarity nikalnaa
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

print()
print(f'Similarity: "software engineer" vs "programmer"       = {sim_1_2:.4f}')
print(f'Similarity: "software engineer" vs "cat sleeping mat" = {sim_1_3:.4f}')