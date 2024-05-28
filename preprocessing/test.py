import json
import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
from preprocessing.GraphEmbedder import GraphEmbedder
from utils import plot



def normalise(number, minimum, maximum):
    return (number - minimum) / (maximum - minimum)

def main():
    with open("out/text_per_tag_katex.json","r") as f:
        text_occurences_per_tag = json.load(f)

    numbers = text_occurences_per_tag["mn"]
    # numbers_to_plot = []
    # for num in numbers:
    #     try:
    #         num = float(num)
    #         numbers_to_plot.append(num)
    #     except:
    #         continue
    # print(len(numbers_to_plot))
    # plot.plot_numbers_distribution(numbers)
    # numers = [0.00017,0.42,3.14,1,800,4e4,9e5,1e12]
    # normalised_num = [normalise(num,0,1e7) for num in numers]
    # print(normalised_num)

    # tree = ET.parse("dataset/cleaned_formulas_katex.xml")
    # root = tree.getroot()
#     texts = ["1.98","209380","1","0.0008","9e7"]
#     embedder = GraphEmbedder()


#     normalised = [embedder.normalise_number(num) for num in texts if embedder.normalise_number(num) is not None]
#     normalised = np.reshape(normalised,(len(normalised),1))

    # numbers = [float(num) for num in texts]
    # number_tensor = torch.tensor(numbers, dtype=torch.float32)
#     linear = SimpleNN(1,10)
#     output = linear(number_tensor.unsqueeze(0))  # Add batch dimension
#     output = linear(torch.tensor(normalised,dtype=torch.float32))
#     print(normalised)
#     print(output)


    
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

#     mi_texts = text_occurences_per_tag["mi"]
#     mi_dim = 10
#     mi_count = len(mi_texts) + 1 # to take into account the unknown
#     unk_token_id = 0

#     mi_embedding = nn.Embedding(mi_count,mi_dim,sparse=True)


#     mi_text_to_id = {text: idx for idx, text in enumerate(mi_texts)}

#     text_ids = [mi_text_to_id.get(text, unk_token_id) for text in texts ]
#     mi_vec = mi_embedding(torch.tensor(text_ids))


#     print("Embedding Table: ",mi_text_to_id)
#     print(f"Text {texts} is found in the table at id {text_ids}")
#     print(f"Torch vector is {mi_vec}")


    # hashed = hash("j") % mi_count
    # mi_vec = mi_embedding(torch.tensor([hashed]))

    # print(f"Text {text} was hashed into id {hashed}")
    # print(f"Torch vector is {mi_vec}")

    # # Step 1: Define the Vocabulary
    # words = ["this", "is", "an", "example", "and", "it", "could", "be", "indefinite", "", "."]
    # vocab = {word: idx for idx, word in enumerate(words)}
    # inv_vocab = {idx: word for word, idx in vocab.items()}  # Inverse vocabulary to map indices back to words
    # vocab_size = len(vocab)
    # embedding_dim = 3  # You can choose the dimensionality of the embeddings

    # # Step 2: Convert Words to Indices
    # indices = [vocab[word] for word in words]

    # # Step 3: Create the Embedding Layer
    # embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    # # Convert indices to a tensor
    # indices_tensor = torch.tensor(indices, dtype=torch.long)

    # # Step 4: Embed the Indices
    # embedded = embedding_layer(indices_tensor)

    # # Recuperate and transform to strings
    # embedded_list = embedded.tolist()  # Convert tensor to list for easier manipulation
    # embedded_str = [str(vec) for vec in embedded_list]  # Convert each embedded vector to string

    # # Map Indices Back to Words
    # recovered_words = [inv_vocab[idx] for idx in indices]

    # print("Vocabulary:", vocab)
    # print("Indices Tensor:", indices_tensor)
    # print("Embedded Vectors:\n", embedded)
    # print("Embedded Vectors as List of Strings:\n", embedded_str)
    # print("Recovered Words:", recovered_words)

    # # --------------------
    # # Example maximum and minimum values for normalization
    # min_val = 1e-5
    # max_val = 1e5

    # def normalize_number(number, min_val, max_val):
    #     return (number - min_val) / (max_val - min_val)


    # # Example large number
    # large_number = 17 * 10**2
    # normalized_number = normalize_number(large_number, min_val, max_val)
    # normalized_number_tensor = torch.tensor([normalized_number], dtype=torch.float32)

    # print(f"Initial value: {large_number}")
    # # Check the normalized value
    # print(f"Normalized value: {normalized_number_tensor.item()}")

    # class SimpleNN(nn.Module):
    #     def __init__(self, input_dim, output_dim):
    #         super(SimpleNN, self).__init__()
    #         self.linear = nn.Linear(input_dim, output_dim)
        
    #     def forward(self, x):
    #         return self.linear(x)

    # # Initialize the network
    # input_dim = 1
    # output_dim = 4  # Example output dimension
    # model = SimpleNN(input_dim, output_dim)

    # # Use the normalized number as input
    # # output = model(normalized_number_tensor.unsqueeze(0))  # Add batch dimension
    # output = model(torch.tensor([large_number],dtype=torch.float32))
    # print(f"Output: {output}")


    # # --------------------


    # # m = nn.Linear(1, 5)
    # # input = torch.tensor([[0.001],[19882]],dtype=torch.float32)
    # # output = m(input)
    # # print("input: ",input)
    # # print("output: ",output)

