import json
import unicodedata
from matplotlib import pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET
import html
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

def count_text_occurences_per_tag(xml_path="dataset/cleaned_formulas_katex.xml", debug=False):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    embedding_table = {tag:dict() for tag in MATHML_TAGS}

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
    
    # Trasform to dict of lists then save it
    # texts_per_tag = {key: list(value) for key,value in embedding_table.items()}
    save.json_dump("out/text_per_tag_katex.json",embedding_table)

    # Plot occurences per tag
    plot.plot_text_frequency_per_tag("out/text_per_tag_katex.json")
    


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