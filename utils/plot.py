import json
import math
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats
import seaborn as sns
from scipy.stats import norm, gaussian_kde

def plot_labels_frequency():
    with open("out/xml_texts.json",mode="r") as f:
        texts_count = json.load(f)
    
    with open("out/xml_tags.json",mode="r") as f:
        tags_count = json.load(f)

    # Remove empty text as a stat
    texts_count.pop("")

    # Sort texts based on their frequency
    sorted_texts = sorted(texts_count.items(), key=lambda x: x[1], reverse=True)
    sorted_tags = sorted(tags_count.items(), key=lambda x: x[1], reverse=True)


    
    # Select top 50 texts and their frequencies
    top_texts = sorted_texts[:30]
    x_label = [label for label, _ in top_texts]
    y_label = [count for _, count in top_texts]

    x_tag = [tag for tag,_ in sorted_tags]
    y_tag = [count for _,count in sorted_tags]

    # Print some stats
    print("Most frequent texts: ",top_texts)


    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    family=['DejaVu Sans']

    # Plot the data
    axs[0].bar(x_label, y_label)
    axs[0].set_title('Top 30 most Frequent texts')
    axs[0].set_xlabel('texts',family=family)
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x')

    axs[1].bar(x_tag, y_tag)
    axs[1].set_title('Most Frequent Tags')
    axs[1].set_xlabel('Tags')
    axs[1].set_ylabel('Count')
    axs[1].tick_params(axis='x', rotation=45)


    plt.tight_layout()
    plt.savefig('out/label_freqs.jpg', format='jpg',dpi=300)


def plot_text_frequency_per_tag(text_per_tag_path="out/text_per_tag.json"):
    with open(text_per_tag_path,mode="r") as f:
        texts_per_tag = json.load(f)

    # with open(texts_count_path,mode="r") as f:
    #     texts_count = json.load(f)

    tags_to_plot = {tag:{text:count for text,count in texts.items() if text != ""} for tag,texts in texts_per_tag.items() if len(texts)>1}
    

    
    print("Sorting texts per occurence for each tag...")
    sorted_texts_per_tag = {tag:sorted(texts.items(), key=lambda x: x[1], reverse=True) for tag, texts in tags_to_plot.items()}    
    # top_texts = {tag:texts[:30] for tag,texts in sorted_texts_per_tag.items()}

    fig, axs = plt.subplots(2, 2, figsize=(16,9))
    fig.suptitle("Text occurences per tag with katex conversion", fontsize=16)
    family=['DejaVu Sans']

    # Iterate over each subplot
    for i in range(2):
        for j in range(2):
            ax = axs[i, j]  # Access the current Axes object
            tag, sorted_texts = list(sorted_texts_per_tag.items())[i * 2 + j]
            top_texts = sorted_texts[:30]
            x_label = [label for label, _ in top_texts]
            y_label = [count for _, count in top_texts]


            ax.bar(x_label, y_label)
            ax.set_title(f'Top 30 most Frequent texts in <{tag}>')
            ax.set_xlabel('texts',family=family)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x')
    
    plt.tight_layout()
    plt.savefig('out/text_freqs_per_tag_katex.jpg', format='jpg',dpi=400)
    print("Saved plot")


def plot_graph(G, name="graph"):
    # Draw the graph using NetworkX's built-in drawing functions
    plt.figure()
    pos = nx.spring_layout(G)  # Compute graph layout

    texts = {n: lab["tag"] +"_"+ lab["text"] if lab["text"] else lab["tag"] for n,lab in G.nodes(data=True)}

    nx.draw_networkx_nodes(G, pos=pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos=pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos=pos, labels=texts,font_size=10, font_family='sans-serif')

    plt.axis('off')

    # Show the graph
    plt.title('MathML Structure Graph')
    plt.savefig(f'out/{name}.jpg', format='jpeg', dpi=300) 


def plot_numbers_distribution(number_occurences,name="distrib"):

    numbers = []
    # occurrences = []

    for key,value in number_occurences.items():
        try:
            num = float(key)
            if num <= 1e15:
                numbers.extend([num]*value)
        except:
            continue
    
    numbers = np.array(numbers)
    # Calculate summary statistics
    mean = np.mean(numbers)
    median = np.median(numbers)
    std_dev = np.std(numbers)
    percentiles = np.percentile(numbers, [1,25, 50, 75,99])
    print(percentiles)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.logspace(-5,15,num=150,base=10)

    # Histogram with KDE
    # sns.histplot(numbers, bins=50, kde=True, color='blue', ax=ax)
    ax.hist(numbers,bins=bins,color="blue",alpha=0.7)

    # Add mean and median lines
    ax.axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean}')
    ax.axvline(median, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median}')

    # Add standard deviation lines
    ax.axvline(mean - std_dev, color='purple', linestyle='--', linewidth=1, label=f'Std Dev: {std_dev}')
    ax.axvline(mean + std_dev, color='purple', linestyle='--', linewidth=1)

    # Add percentiles
    ax.axvline(percentiles[0], color='orange', linestyle='-.', linewidth=1.5, label=f'1st Percentile: {percentiles[0]:.2f}')
    ax.axvline(percentiles[4], color='orange', linestyle='-.', linewidth=1.5, label=f'99th Percentile: {percentiles[4]:.2f}')


    # Set x-scale to log
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Distribution of Numbers')
    ax.set_xlabel('Numbers')
    ax.set_ylabel('Frequency')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'out/{name}.jpg', format='jpeg', dpi=300) 


def plot_loss_graph(val_losses,train_losses, dir_path):

    num_epochs = len(train_losses)

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig(f'{dir_path}/loss_graph.png')

def plot_training_graphs(history,dir_path):

    train_losses, val_losses, aucs, aps = history["loss"], history["val_loss"], history["auc"], history["ap"]

    epochs = range(1, len(train_losses) + 1)

    # Create a figure with subplots
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    # fig.suptitle('Training Metrics Over Epochs')

    # Plot Train and Validation Losses
    ax1 = ax[0]
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs, val_losses, label='Validation Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train & Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # # Remove the top-right empty plot
    # fig.delaxes(ax[0, 1])

    # Plot AUC
    ax2 = ax[1]
    ax2.plot(epochs, aucs, label='AUC', color='green')
    ax2.plot(epochs, aps, label='AP', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('AUC & AP scores')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{dir_path}/training_metrics.png')