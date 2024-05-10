from datasets import load_dataset
from latex2mathml import converter
import xml.etree.ElementTree as ET
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from latex2mathml.exceptions import InvalidAlignmentError, DoubleSuperscriptsError, DenominatorNotFoundError
from tqdm import tqdm


# Dataset of formulas : https://huggingface.co/datasets/OleehyO/latex-formulas
# There are two datasets: raw_formulas and cleaned_formulas(This dataset has 550K formula-image pairs)
# "We scraped approximately 1 million LaTeX formula image-text pairs from arxiv that were uncleaned and without 
# text segmentation to create the raw_formulas dataset. After cleaning the raw_formulas dataset and integrating 
# it with the im2latex-100K dataset, we obtained the cleaned_formulas dataset, which has 550K formula-image pairs."


# data = load_dataset("OleehyO/latex-formulas", "raw_formulas",trust_remote_code=True) 
# data = load_dataset("OleehyO/latex-formulas", "cleaned_formulas",trust_remote_code=True) 
conversion_stats = {
    "success":0,
    "align":0,
    "superscript":0,
    "denominator":0,
    "dunno":0,
}

data = load_dataset("OleehyO/latex-formulas", "cleaned_formulas",trust_remote_code=True) 

# Initialize a string to store all XML documents
root = ET.Element("formulas")

# Iterate over all equations in the dataset
for i, formula in enumerate(tqdm(data["train"])):
    latex_formula = formula["latex_formula"]

    # Convert LaTeX formula to MathML
    try:
        mathml_element = converter.convert_to_element(latex_formula)
        conversion_stats["success"] += 1
    except InvalidAlignmentError:        
        try:
            cleaned = latex_formula.replace(r"\begin{align*}","").replace(r"\end{align*}","")
            mathml_element = converter.convert_to_element(cleaned)
        except Exception as e:
            # print(f"Error removing alginment at index {i}: {latex_formula} -- {e}")
            conversion_stats["align"] += 1
    except DoubleSuperscriptsError:
        # print(f"Error removing DoubleSuperscript at index {i}")
        conversion_stats["superscript"] += 1

    except DenominatorNotFoundError:
        # print(f"Error converting formula at index {i}")
        conversion_stats["denominator"] += 1
    
    except Exception as e:
        # print(f"Exception {e}")
        conversion_stats["dunno"] += 1

    root.append(mathml_element)
print(f"The latex formulas were converted, here are the stats : {conversion_stats}")

tree = ET.ElementTree(root)
tree.write("dataset/equations.xml", encoding="utf-8", xml_declaration=True)
