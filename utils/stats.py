import json
from matplotlib import pyplot as plt
import networkx as nx
import xml.etree.ElementTree as ET

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

def xml_occurences(xml_path="dataset/cleaned_formulas.xml", debug=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_tags = {}
    xml_texts = {}

    def count_in_element(element):        
        
        if "math" in element.tag:
            tag = rn(element.tag)
            text = element.text
            xml_tags[tag] = xml_tags.get(tag, 0) + 1
            xml_texts[text] = xml_texts.get(text, 0) + 1

        for child in element:
            tag = rn(child.tag)
            text = child.text

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
    # plot.plot_labels_frequency()

        
def rn(x):
    """Remove Namespace"""
    return x.replace(r"{http://www.w3.org/1998/Math/MathML}", "")