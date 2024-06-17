import json
import os
import random
import unicodedata
from matplotlib import pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET
import html
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler
import torch
from tqdm import tqdm
from utils import save, plot

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


def xml_occurences(xml_path="dataset/equations.xml", debug=False):


    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_tags = {}
    xml_texts = {}

    def count_in_element(element):        
        
        if "math" in element.tag:
            tag = rn(element.tag)
            text = "" if element.text is None else element.text
            xml_tags[tag] = xml_tags.get(tag, 0) + 1
            xml_texts[text] = xml_texts.get(text, 0) + 1

        for child in element:
            tag = rn(child.tag)
            text = "" if child.text is None else child.text
            

            xml_tags[tag] = xml_tags.get(tag, 0) + 1
            xml_texts[text] = xml_texts.get(text, 0) + 1

            children = [x for x in child]
            if children:
                count_in_element(child)



    # iterate over each XML equation
    for i, formula in enumerate(tqdm(root,desc="Counting occurences",unit="equations")):
        if debug and i>= 10:
            break

        # Run recursive function
        count_in_element(formula)
    
    print("Number of different tags: ", len(xml_tags.keys()))
    print("Number of different labels: ",len(xml_texts.keys()))

    save.json_dump("out/xml_tags.json",xml_tags)
    save.json_dump("out/xml_texts.json",xml_texts)
    plot.plot_labels_frequency()

def count_text_occurences_per_tag(xml_path="dataset/raw/cleaned_formulas_katex.xml", debug=False):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    embedding_table = {tag:dict() for tag in MATHML_TAGS}
    vocab_table = {"":0,"<unk>":1}

    bad_things = {"numbers":0}

    def find_in_element(element):    
        
        if "math" in element.tag:
            tag = rn(element.tag)
            text = "" if element.text is None else clean_text(element.text)
            embedding_table[tag][text] = embedding_table[tag].get(text,0) + 1


        for child in element:
            tag = rn(child.tag)
            text = "" if child.text is None else clean_text(child.text)

            if tag=="mn":
                try:
                    number = float(text)
                except:
                    bad_things["numbers"] +=1

            embedding_table[tag][text] = embedding_table[tag].get(text,0) + 1
            if text not in vocab_table:
                vocab_table[text] = len(vocab_table)

            children = [x for x in child]
            if children:
                find_in_element(child)
    
    # iterate over each XML equation
    for i, formula in enumerate(tqdm(root,desc="Counting occurences",unit="equations")):
        if debug and i>= 10000:
            break

        # Run recursive function
        find_in_element(formula)
    
    for tag,values in embedding_table.items():
        if len(values) > 1:
            print(f"{tag} : {len(values)} - examples : {list(values)[0:5] if len(values)>5 else list(values)}")
    print(bad_things)
    
    save.json_dump("out/vocab_texts_katex.json",vocab_table)

    # Trasform to dict of lists then save it
    # texts_per_tag = {key: list(value) for key,value in embedding_table.items()}
    # save.json_dump("out/text_per_tag_katex.json",embedding_table)

    # # Plot occurences per tag
    # plot.plot_text_frequency_per_tag("out/text_per_tag_katex.json")
    
def test_different_feature_scalings():
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open("out/text_per_tag_katex.json","r") as f:
        text_occurences_per_tag = json.load(f)

    num_values = text_occurences_per_tag["mn"]
    num_list = []
    for key, value in num_values.items():
        try:
            key = float(key)
            if key <= 1e6:
                num_list += [key] * value
        except:
            continue

    # numbers_occ = [[float(key)]*value for key, value in numbers.items() if float(key)!= None]
    # flat_numbers = [x for xs in numbers_occ for x in xs]
    num_vec = np.array(num_list,dtype=np.float32)
    np.random.shuffle(num_vec)
    numbers = num_vec.reshape(-1,16)

    # print(numbers[0])

    dtype = torch.float32
    # numbers = np.array([[0, 1e-2, 2e-1, 1],
    #                     [3.14, 5e3, 5e2, 0.01]],dtype=np.float32)

    # min-max normalisation
    normed = normalize(numbers)

    # Mean normalisation
    mean_val = np.mean(numbers)
    range_val = np.ptp(numbers)  # Range is max - min
    mean_normed = mean_normalize(numbers, mean_val, range_val)

    # Log transformation
    log_transformed = log_transform(numbers)

    # robust scaling
    scaler = RobustScaler()
    scaled = scaler.fit_transform(numbers)
    # print(numbers[12],scaled[12])
    reverse_scaled = scaler.inverse_transform(scaled)

    # box-cox transfo
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    power_transformed = pt.fit_transform(numbers)
    reverse_powered = pt.inverse_transform(power_transformed)


    big_numbers = {
        "original":numbers,
        "min-max_normalisation":normed,
        "mean_normalisation":mean_normed,
        "log_transform":log_transformed,
        "robust_scaling":scaled,
        "power_transformation":power_transformed
    }

    plot.plot_multiple_distributions(big_numbers)


def extract_data_from_search(path="data/ray_results/GAE_search_Pfhaler"):
    trial_histories = []
    name = path.split("/")[-1]
    try:
        trial_paths = [os.path.join(path,trial) for trial in os.listdir(path) if os.path.isdir(os.path.join(path,trial))]
        for _path in trial_paths:
            files_list = os.listdir(_path)
            if "progress.csv" in files_list:
                params_dir = os.path.join(_path,"params.json")
                progress_dir = os.path.join(_path,"progress.csv")

                # Load params
                with open(params_dir,"r") as f:
                    params = json.load(f)
                
                # Load csv
                results = pd.read_csv(progress_dir)
                metrics_head = results.keys()[:4]

                # generate history dict
                history = {key:list(results[key]) for key in metrics_head}
                history["params"] = params

                trial_name = _path.split("/")[-1].split("_")[2]
                history["trial_name"] = trial_name

                trial_histories.append(history)

    except Exception as e:
        print(f"Couldn't extract data due to {e}")
    
    print(f" Found {len(trial_histories)} trials")

    with open(os.path.join("trained_models",name,"history.json"),"r") as f:
        best_trial= json.load(f)

    try:
        plot.plot_hyperparam_search(trial_histories,name,best_trial=best_trial)
    except Exception as e:
        print(f"Couldn't plot the HP search figure because of {e}")

    try:
        save.json_dump(f"trained_models/{name}/all_histories.json",trial_histories)
    except Exception as e:
        print(f"couldn't save the histories because of {e}")



def decode_xml_entities(text):
    return html.unescape(text)

def normalize_unicode(text):
    return unicodedata.normalize('NFKC', text)

def clean_text(text):
    text = decode_xml_entities(text)
    # text = normalize_unicode(text) # Maybe not?

    text = text.replace('\u00a0',' ').strip()
    return text

def rn(x):
    """Remove Namespace"""
    return x.replace(r"{http://www.w3.org/1998/Math/MathML}", "")

def log_transform(number):
    return np.log10(number + 1e-7)
    # return torch.log(torch.tensor(number + 1e-7))

def inverse_log_transform(tensor):
    return torch.exp(tensor) - 1e-7

def mean_normalize(number, mean, range_val):
    return (number - mean) / range_val

def inverse_mean_normalize(normalized_number, mean, range_val):
    return (normalized_number * range_val) + mean


def normalize(number,min_val=0,max_val=1e6):
    return (number - min_val) / (max_val - min_val)