from datasets import load_dataset
from latex2mathml import converter
import xml.etree.ElementTree as ET
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt


# Dataset of formulas : https://huggingface.co/datasets/OleehyO/latex-formulas
# There are two datasets: raw_formulas and cleaned_formulas(This dataset has 550K formula-image pairs)
# "We scraped approximately 1 million LaTeX formula image-text pairs from arxiv that were uncleaned and without 
# text segmentation to create the raw_formulas dataset. After cleaning the raw_formulas dataset and integrating 
# it with the im2latex-100K dataset, we obtained the cleaned_formulas dataset, which has 550K formula-image pairs."



# data = load_dataset("OleehyO/latex-formulas", "raw_formulas",trust_remote_code=True) 
data = load_dataset("OleehyO/latex-formulas", "cleaned_formulas",trust_remote_code=True) 

example = data["train"][101]
# print(example["latex_formula"])
Image.Image.save(example["image"],"out/example.jpeg")

mathml_output = converter.convert(example["latex_formula"])
test = converter.convert_to_element(example["latex_formula"])

tree = ET.ElementTree(test)
tree.write("out/test.xml", encoding="utf-8", xml_declaration=True)


