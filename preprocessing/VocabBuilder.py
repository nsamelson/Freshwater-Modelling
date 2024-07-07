import os
import random
import unicodedata
import json
import xml.etree.ElementTree as ET
import html
from tqdm import tqdm

from config import MATHML_TAGS, ROOT_DIR


VOCAB_TYPES = [
    "combined",
    "split",
    "concat"
]

class VocabBuilder():
    def __init__(self, xml_name: str, vocab_type="combined", reload_vocab=False, reload_xml_elements=False, debug=False, ) -> None:
        self.xml_name = xml_name
        self.reload_vocab = reload_vocab
        self.reload_xml_elements = reload_xml_elements
        self.debug = debug
        self.vocab_type = vocab_type if vocab_type in VOCAB_TYPES else "combined"

        root_path = os.path.join(ROOT_DIR,"data/pre_processed")

        self.dir_path = os.path.join(root_path,xml_name)
        self.xml_path = os.path.join(root_path,xml_name,"raw/equations.xml")
        self.vocab_path = os.path.join(root_path,xml_name,"vocab.json")
        self.element_dict_path = os.path.join(root_path,xml_name,"xml_elements.json")

        self.xml_elements = {tag:dict() for tag in MATHML_TAGS}
        self.vocab_table = {} # "":0,"<unk>":1

        if not os.path.exists(self.xml_path):
            raise Exception("No XML found!")

        # checks if xml_elements has been processed
        if not os.path.exists(self.element_dict_path) or reload_xml_elements:
            self.process_xml_elements()
        else:
            self.load_xml_elements()
        
        # checks if vocab has been processed
        if not os.path.exists(self.vocab_path) or reload_vocab:
            self.process_vocab()
        else:
            self.load_vocab()
    
    def process_vocab(self):
        def index_vocab(values, index):
            sorted_vals = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
            indexed_vals = {}
            for k in sorted_vals.keys():
                if k != "":
                    indexed_vals[k] = index
                    index += 1
            return indexed_vals

        if self.vocab_type == "combined":
            self.vocab_table = {"":0,"<unk>":1} 

            # Flatten           
            flattened_dict = {}            
            for _, values in self.xml_elements.items():
                flattened_dict.update(values)
            
            # put indices for each value, in descending order of freqs
            self.vocab_table.update(index_vocab(flattened_dict,2))

        elif self.vocab_type == "concat":
            self.vocab_table = {"":0,"<unk>":1} 
            index = 2
            # Flatten           
            flattened_dict = {}            
            for element, values in self.xml_elements.items():
                concat_values = {"_".join([element,key]):value  for key,value in values.items()}
                flattened_dict.update(concat_values)
                self.vocab_table[element] = index
                index +=1

            # put indices for each value, in descending order of freqs
            self.vocab_table.update(index_vocab(flattened_dict,index))

        elif self.vocab_type == "split":
            for element, values in self.xml_elements.items():
                if len(values.values())<= 1:
                    continue                
                self.vocab_table[element] = {"":0,"<unk>":1} # if element != "mn" else {}
                self.vocab_table[element].update(index_vocab(values,2))
        
        
        # Save stuff
        print("Saving Vocab...")
        with open(self.vocab_path,"w+") as f:
            json.dump(self.vocab_table,f)
        
        

    def process_xml_elements(self):
        print("Loading XML...")
        tree = ET.parse(self.xml_path)
        root = tree.getroot()       

        def find_in_element(element):         

            # First element   
            if "math" in element.tag:
                tag = rn(element.tag)
                text = "" if element.text is None else clean_text(element.text)
                self.xml_elements[tag][text] = self.xml_elements[tag].get(text,0) + 1

            for child in element:
                tag = rn(child.tag)
                text = "" if child.text is None else clean_text(child.text)

                self.xml_elements[tag][text] = self.xml_elements[tag].get(text,0) + 1

                children = [x for x in child]
                if children:
                    find_in_element(child)

        # iterate over each XML equation
        for i, formula in enumerate(tqdm(root,desc="Generating vocab",unit=" equations",total=len(root))):
            if self.debug and i>= 10000:
                break
            # Run recursive function
            find_in_element(formula)
        
        print("Saving xml elements...")
        with open(self.element_dict_path,"w+") as f:
            json.dump(self.xml_elements,f)

    def load_vocab(self):
        with open(self.vocab_path,"r") as f:
            self.vocab_table = json.load(f)
    
    def load_xml_elements(self):
        with open(self.element_dict_path,"r") as f:
            self.xml_elements = json.load(f)

    def __len__(self):
        return len(self.vocab_table)

    # def __getitem__(self, idx):
    #     return self.vocab_table[idx] # might be confusing!!!
    


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



if __name__=="__main__":
    # Usage example
    print("starting stuff")
    vocab  = VocabBuilder("default", vocab_type="combined", reload_vocab=True, reload_xml_elements=False, debug=False)
    vocab.process_vocab()
    # print(vocab.vocab_table)
    # print(f"The latex formulas were converted, here are the stats : {dataset.stats}")
    # print(ET.tostring(dataset[0], encoding='unicode'))