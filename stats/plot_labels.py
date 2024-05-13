import json
from matplotlib import pyplot as plt


def plot_labels_frequency():
    with open("out/xml_labels_count.json",mode="r") as f:
        labels_count = json.load(f)
    
    with open("out/xml_tags_count.json",mode="r") as f:
        tags_count = json.load(f)

    # Sort labels based on their frequency
    sorted_labels = sorted(labels_count.items(), key=lambda x: x[1], reverse=True)
    sorted_tags = sorted(tags_count.items(), key=lambda x: x[1], reverse=True)
    
    # Select top 50 labels and their frequencies
    top_labels = sorted_labels[:30]
    x_label = [label for label, _ in top_labels]
    y_label = [count for _, count in top_labels]

    x_tag = [tag for tag,_ in sorted_tags]
    y_tag = [count for _,count in sorted_tags]


    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # Plot the data
    axs[0].bar(x_label, y_label)
    axs[0].set_title('Top 30 most Frequent Labels')
    axs[0].set_xlabel('Labels')
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x', rotation=45, )

    axs[1].bar(x_tag, y_tag)
    axs[1].set_title('Most Frequent Tags')
    axs[1].set_xlabel('Tags')
    axs[1].set_ylabel('Count')
    axs[1].tick_params(axis='x', rotation=45)


    plt.tight_layout()
    plt.savefig('out/label_freqs.jpg', format='jpg',dpi=300)
