import json
import math
import os
from matplotlib import cm, pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from scipy.stats import norm, gaussian_kde
from torch_geometric.utils.convert import to_networkx, from_networkx

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


MATHML_TAGS = [
    "maction",
    "math",
    "menclose",
    "merror", 
    "mfenced",
    "mfrac", 
    "mglyph", 
    "mi", 	
    "mlabeledtr", 
    "mmultiscripts", 
    "mn",
    "mo",
    "mover", 	
    "mpadded", 	
    "mphantom", 	
    "mroot", 	
    "mrow", 
    "ms", 	
    "mspace",
    "msqrt",
    "mstyle",
    "msub",
    "msubsup",  
    "msup",
    "mtable",
    "mtd",
    "mtext",
    "mtr",
    "munder",
    "munderover",
    "semantics", 
]


def plot_graph(G, name="graph"):
    # Draw the graph using NetworkX's built-in drawing functions
    plt.figure()
    pos = nx.spring_layout(G)  # Compute graph layout

    try:
        texts = {n: lab["tag"] +"_"+ lab["text"] if lab["text"] else lab["tag"] for n,lab in G.nodes(data=True)}
    except:
        texts = {i:text["x"] for i,text in G.nodes(data=True)}

    nx.draw_networkx_nodes(G, pos=pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos=pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos=pos, labels=texts,font_size=10, font_family='sans-serif')

    plt.axis('off')

    dir = os.getcwd()
    path = os.path.join(dir,"out",f"{name}.jpg")

    # Show the graph
    plt.title('MathML Structure Graph')
    plt.savefig(path, format='jpeg', dpi=300) 

def plot_from_pyg(pyg,name="pyg"):
    
    G = to_networkx(pyg)
    plot_graph(G,name)


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


def plot_training_history(history, dir_path):

    metrics = set([key.split("_")[1] for key in history.keys() if "_" in key])

    def plot_loss_graph(val_data,train_data, metric):

        num_epochs = len(val_data)

        plt.figure(figsize=(12,8))
        plt.plot(range(1, num_epochs + 1), train_data, label=f'Train {metric}', color='blue')
        plt.plot(range(1, num_epochs + 1), val_data, label=f'Validation {metric}', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Train and Validation {metric}')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{dir_path}/{metric}_history.png')

    for metric in metrics:
        
        train_metric = "_".join(["train",metric])
        val_metric = "_".join(["val",metric])

        train_data = history[train_metric] if train_metric in history else history[metric]
        val_data = history[val_metric] 
        plot_loss_graph(val_data, train_data, metric)



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
    train_losses, val_losses, aucs, aps = history["loss"], history["val_loss"], history["val_auc"], history["val_ap"]

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

def plot_multiple_distributions(arrays_dict={}):


    # Create a figure with 6 subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    x,y = 0,0
    for title, array in arrays_dict.items():
        flattened_array = array.flatten()

        max_val = np.max(flattened_array)
        min_val = np.min(flattened_array)  
        if title == "mean_normalisation":
            flattened_array = flattened_array - min_val
        
        nonzero_data = flattened_array[flattened_array > 0]
        log_values = np.log10(nonzero_data)
        smallest_exponent = np.min(np.floor(log_values))
        biggest_exponent = np.max(np.floor(log_values)) + 1


        mean = np.mean(flattened_array)
        median = np.median(flattened_array)
        std_dev = np.std(flattened_array)
        percentiles = np.percentile(flattened_array, [25, 75])

        axs[x,y].axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean:.2e}')
        axs[x,y].axvline(median, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median:.2e}')
        axs[x,y].axvline(mean - std_dev, color='purple', linestyle='--', linewidth=1, label=f'Std Dev: {std_dev:.2e}')
        axs[x,y].axvline(mean + std_dev, color='purple', linestyle='--', linewidth=1)
        axs[x,y].axvline(percentiles[0], color='orange', linestyle='-.', linewidth=1.5, label=f'Q1: {percentiles[0]:.2e}')
        axs[x,y].axvline(percentiles[1], color='orange', linestyle='-.', linewidth=1.5, label=f'Q3: {percentiles[1]:.2e}')

        if title in ["original","robust_scaling","min-max_normalisation","mean_normalisation"]:

            # if min_val >= 0:
            bins = np.logspace(smallest_exponent,biggest_exponent,num=100,base=10)
            axs[x,y].set_xscale('log')
        
        else:
            bins = 100
        
        axs[x,y].hist(flattened_array, bins=bins, alpha=0.75, color='blue')
        

        axs[x,y].set_title(title)
        axs[x,y].set_ylabel('occurences')
        axs[x,y].set_yscale('log')
        axs[x,y].set_xlabel('value')

        axs[x,y].legend()
        if x < 1:
            x+=1
        else:
            x=0
            y+=1

    plt.suptitle('Different feature scaling methods',fontsize=16)
    # Adjust layout
    plt.tight_layout()
    # Save the figure
    plt.savefig('out/transformation_distributions.png')


# Function to dynamically rename classes by extracting the last part of the class name
def rename_classes_dynamic(df, class_columns):
    for col in class_columns:
        df[col] = df[col].apply(lambda x: x.split('.')[-1].strip(">'") if isinstance(x, str) and 'class' in x else x)
    return df


def plot_hyperparam_search(dir_path,metrics=["auc", "ap", "val_loss"],scatter_params=["lr"],filter_params=["num_epochs"], max_loss=1.2):
    try:
        with open(os.path.join(dir_path,"all_histories.json"),"r") as f:
            histories = json.load(f)
    except Exception as e:
        print(f"couldn't load histories because of {e}")
        return
    try:
        with open(os.path.join(dir_path,"history.json"),"r") as f:
            best_trial = json.load(f)
    except:
        print(f"No best trial 'history.json' found under {dir_path}")
        best_trial = None
    
    all_params = histories[0]["params"].keys()
    search_params = [param for param in all_params if param not in filter_params]
    processed_data = []

    # Extract the parameters and their corresponding metrics
    for entry in histories:
        params = entry["params"]
        best_run_idx = entry["val_loss"].index(min(entry["val_loss"]))

        # don't print data with loss bigger than the max specified
        if entry["val_loss"][best_run_idx] > max_loss:
            continue
        for metric in metrics:

            processed_data.append({
                "metric": metric,
                "value": entry[metric][best_run_idx],
                **params
            })
    df = pd.DataFrame(processed_data)

    # Identify columns that contain class names and rename them
    class_columns = [col for col in search_params if any(isinstance(val, str) and 'class' in val for val in df[col].unique())]
    df = rename_classes_dynamic(df, class_columns)
    num_cols = len(search_params)

    # Create a figure to plot all subplots
    fig, axes = plt.subplots(len(metrics), num_cols, figsize=(num_cols*3, num_cols))

    # Plot each hyperparameter's metrics
    for i, hyperparameter in enumerate(search_params):
        for j, metric in enumerate(metrics):
            ax = axes[j,i]
            scatter = True if hyperparameter in scatter_params else False

            if scatter:
                sns.scatterplot(x=hyperparameter, y="value", hue="metric", data=df[df["metric"] == metric], ax=ax,)
                ax.set_xscale("log")         
            else:
                sns.boxplot(x=hyperparameter, y="value", hue="metric", data=df[df["metric"] == metric], ax=ax, )


            if best_trial is not None:
                best_param = best_trial["params"][hyperparameter]
                best_param = best_param.split('.')[-1].strip(">'") if isinstance(best_param, str) and 'class' in best_param else best_param
                best_metric_idx = best_trial["val_loss"].index(min(best_trial["val_loss"]))
                best_metric = best_trial[metric][best_metric_idx]
                
                # Add marker for best_param
                ticks = [text.get_text() for text in ax.get_xticklabels()]
                index = ticks.index(str(best_param)) if not scatter else best_param
                ax.scatter(x=[index], y=[best_metric], marker="*", c="r",s=50, zorder=10)


            # Add a single legend outside the plot
            ax.set_xlabel(hyperparameter,fontsize=14) if j == len(metrics) - 1 else ax.set_xlabel("")
            ax.set_ylabel(metric,fontsize=16) if i==0 else ax.set_ylabel("")                        
            ax.legend().set_visible(False)
    
    plt.suptitle(f'Hyperparameter search plot with {len(histories)} trials',fontsize=22)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(dir_path,'hyperparameter_boxplots.png'))
        
def plot_history_from_search(model_name="concat_search_var"):
    with open(os.path.join("trained_models", model_name, "all_histories.json"), "r") as f:
        histories = json.load(f)

    history_per_metric = {}
    params_per_training = {}
    metrics_to_watch = ["val_loss", "val_acc", "val_sim"]
    col_type = "train_edge_features"
    row_type = "embed_method"
    cat_type = "concat_dim"

    max_epochs = 15
    categories = set()
    y_lim = {}

        # Initialize structures to store data
    categories = set()
    history_data = {metric: [] for metric in metrics_to_watch}
    y_lim = {metric: {} for metric in metrics_to_watch}

    # Process the histories
    for training in histories:
        params = training["params"]
        col = params[col_type]
        row = params[row_type]
        cat = params[cat_type]
        categories.add(cat)

        for metric, data in training.items():
            if metric in metrics_to_watch:
                data = np.array(data[:max_epochs])
                history_data[metric].append((col, row, cat, data))
                if (row, cat) not in y_lim[metric]:
                    y_lim[metric][(row, cat)] = {"data": [data]}
                else:
                    y_lim[metric][(row, cat)]["data"].append(data)
                    
    # Calculate Q1 and Q3 for each row and category
    for metric in y_lim:
        for key in y_lim[metric]:
            data = np.array(y_lim[metric][key]["data"])
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            y_lim[metric][key] = {"q1": q1, "q3": q3}

    def plot_metric_graph(metric):
        data = history_data[metric]
        cols = list(set([item[0] for item in data]))
        rows = list(set([item[1] for item in data]))
        num_cols = len(cols)
        num_rows = len(rows)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan', 'tab:pink']
        category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), sharex=True)

        row_limits = {row: {"min": float('inf'), "max": float('-inf')} for row in rows}

        # First pass to find y-limits per row
        for (col, row, cat, y) in data:
            row_limits[row]["min"] = min(row_limits[row]["min"], y.min())
            row_limits[row]["max"] = max(row_limits[row]["max"], y.max())

        for (col, row, cat, y) in data:
            i = rows.index(row)
            j = cols.index(col)
            ax = axs[i, j] if num_rows > 1 and num_cols > 1 else (axs[i] if num_rows > 1 else (axs[j] if num_cols > 1 else axs))

            color = category_colors[cat]
            x = range(1, max_epochs + 1)
            ax.plot(x, y, label=f"{cat_type}: {cat}", color=color)
            q1 = y_lim[metric][(row, cat)]["q1"]
            q3 = y_lim[metric][(row, cat)]["q3"]
            ax.fill_between(x, q1, q3, alpha=0.05, color=color)

            ax.set_ylim(row_limits[row]["min"], row_limits[row]["max"])

            if i == 0:
                ax.set_title(f"{col_type}: {col}")
            if j == 0:
                ax.set_ylabel(f"{row_type}: {row}")
            ax.set_xlabel('Epoch')
            ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())
        plt.suptitle(f'History {metric} per {col_type} and {row_type}', fontsize=14)
        plt.tight_layout()  # Adjust layout to fit suptitle
        plt.savefig(f'trained_models/{model_name}/history_{metric}_per_{col_type}_and_{row_type}.png')

    for metric in metrics_to_watch:
        plot_metric_graph(metric)
    # print(history_per_metric["trial_name"])
    # print(params_per_training[0])

def plot_heatmap(model_name):

    data_path = os.path.join("trained_models",model_name,"all_progress.csv")
    df = pd.read_csv(data_path)

    colormap = "inferno"
    

    print("DataFrame head:")
    print(df.head())
    print("\nDataFrame columns:")
    print(df.columns)
    

    # Heatmap of Validation Loss by Embed Method and Edge Features
    heatmap_data = df.pivot_table(index='embed_method', columns='train_edge_features', values='val_acc', aggfunc='mean')
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_data, annot=True, cmap=colormap)
    plt.title('Validation Accuracy by Embed Method and Edge Features')
    plt.savefig(f'trained_models/{model_name}/heatmap.png')
    plt.close()

    # Boxplot of Validation Loss by Tag Dimension
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='concat_dim', y='val_loss', data=df)
    plt.title('Validation Loss by Tag Dimension')
    plt.savefig(f'trained_models/{model_name}/boxplot.png')
    plt.close()

    # 3D Scatter Plot of Concatenation Dimension, Positional Dimension, and Accuracy
    if "val_acc" in df.columns and "concat_dim" in df.columns and "pos_dim" in df.columns:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(df['val_acc'], df['val_loss'], df['val_sim'], c=df['train_edge_features'], cmap=colormap)
        plt.colorbar(sc)
        ax.set_xlabel('Concat Dim')
        ax.set_ylabel('Pos Dim')
        ax.set_zlabel('Accuracy')
        plt.title('3D Scatter Plot of Concat Dim, Pos Dim, and Accuracy')
        plt.savefig(f'trained_models/{model_name}/scatter.png')
        plt.close()
    else:
        print("Columns for 3D scatter plot are missing in the DataFrame.")


def plot_study(model_name):
    data_path = os.path.join("trained_models",model_name,"all_progress.csv")
    df = pd.read_csv(data_path)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan', 'tab:pink']
    
    # relevant_columns = ['val_loss', "loss", 'training_iteration', 'pos_dim', 'train_edge_features', 'folder']
    # df = df[relevant_columns]
    df['pos_dim'] = df['pos_dim'].fillna('NaN')

    groups = df.groupby(['pos_dim', 'train_edge_features'])

    metrics = [('val_loss','loss'),("val_acc","train_acc"),("val_sim","train_sim")]
    metric_names = ["loss","accuracy","similarity"]
    fig, axs = plt.subplots(1, len(metrics), figsize=(4 * len(metrics),4))

    group_labels = []
    for i, metric in enumerate(metrics):
        for j,((pos_dim, train_edge_features), group) in enumerate(groups):
            # group_label = f'pos: {pos_dim}, ef: {train_edge_features}'
            group_label = "concat"
            group_label += "" if pos_dim =="NaN" else " + pos"
            group_label += " + ef" if train_edge_features else ""

            if group_label not in group_labels:
                group_labels.append(group_label)

            
            axs[i].plot(group['training_iteration'], group[metric[0]], label=group_label, color = colors[j])
            axs[i].plot(group['training_iteration'], group[metric[1]], color = colors[j], linestyle='dashed')
        
        if "loss" not in metric:
            axs[i].set_ylim(0, 1.05)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric_names[i])
        axs[i].set_title(f'Train and val {metric_names[i]}')
        axs[i].grid(True)

    
    handles, labels = [], []
    for j, (label) in enumerate(group_labels):
        handle = plt.Line2D([0], [1], color=colors[j], label=label)
        handles.append(handle)
        labels.append(label)
    axs[0].legend(handles=handles, labels=labels, loc='upper right')

    train_handle = plt.Line2D([0], [0], color='black', linestyle='dashed', label='Training')
    val_handle = plt.Line2D([0], [0], color='black', linestyle='solid', label='Validation')
    handles = [train_handle,val_handle]
    labels = ['Training', 'Validation']
    axs[-1].legend(handles=handles, labels=labels, loc='lower right')


    plt.tight_layout(rect=[0,0, 1, 1])
    # plt.suptitle("Ablation Study",fontsize=16)

    plt.savefig(f'trained_models/{model_name}/loss_acc_sim.png')
    # plt.savefig(f'trained_models/{model_name}/loss_acc_sim.pdf')