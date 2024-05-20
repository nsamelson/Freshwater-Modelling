import torch
import torch.nn as nn

# Step 1: Define the Vocabulary
words = ["this", "is", "an", "example", "and", "it", "could", "be", "indefinite", "", "."]
vocab = {word: idx for idx, word in enumerate(words)}
inv_vocab = {idx: word for word, idx in vocab.items()}  # Inverse vocabulary to map indices back to words
vocab_size = len(vocab)
embedding_dim = 3  # You can choose the dimensionality of the embeddings

# Step 2: Convert Words to Indices
indices = [vocab[word] for word in words]

# Step 3: Create the Embedding Layer
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Convert indices to a tensor
indices_tensor = torch.tensor(indices, dtype=torch.long)

# Step 4: Embed the Indices
embedded = embedding_layer(indices_tensor)

# # Recuperate and transform to strings
# embedded_list = embedded.tolist()  # Convert tensor to list for easier manipulation
# embedded_str = [str(vec) for vec in embedded_list]  # Convert each embedded vector to string

# # Map Indices Back to Words
# recovered_words = [inv_vocab[idx] for idx in indices]

print("Vocabulary:", vocab)
print("Indices Tensor:", indices_tensor)
print("Embedded Vectors:\n", embedded)
# print("Embedded Vectors as List of Strings:\n", embedded_str)
# print("Recovered Words:", recovered_words)

