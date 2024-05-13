from datasets import load_dataset
from latex2mathml import converter
import xml.etree.ElementTree as ET
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from latex2mathml.exceptions import InvalidAlignmentError, DoubleSuperscriptsError, DenominatorNotFoundError
from tqdm import tqdm


def main():
    """
    Download Tex equations, convert to XML and save into XML dataset.
    Dataset of formulas : https://huggingface.co/datasets/OleehyO/latex-formulas
    Page info: "We scraped approximately 1 million LaTeX formula image-text pairs from arxiv that were uncleaned and without 
    text segmentation to create the raw_formulas dataset. After cleaning the raw_formulas dataset and integrating 
    it with the im2latex-100K dataset, we obtained the cleaned_formulas dataset, which has 550K formula-image pairs."
    """
    data = load_dataset("OleehyO/latex-formulas", "raw_formulas",trust_remote_code=True) 
    # data = load_dataset("OleehyO/latex-formulas", "cleaned_formulas",trust_remote_code=True) 

    conversion_stats = {
        "success":0,
        "align":0,
        "superscript":0,
        "denominator":0,
        "dunno":0,
    }

    # Initialize a string to store all XML documents
    root = ET.Element("formulas")

    # Iterate over all equations in the dataset
    for i, formula in enumerate(tqdm(data["train"], desc="Generating XML",unit="equations")):
        latex_formula = formula["latex_formula"]

        # Convert LaTeX formula to MathML
        try:
            # Clean LaTeX formula to remove unnecessary tags
            for tag in [r"\begin{align*}", r"\end{align*}", r"\begin{align}", r"\end{align}", r"\begin{equation*}", r"\end{equation*}", r"\begin{equation}", r"\end{equation}"]:
                latex_formula = latex_formula.replace(tag, "")

            # Convert to MathML
            mathml_element = converter.convert_to_element(latex_formula)
            conversion_stats["success"] += 1

        except InvalidAlignmentError:
            conversion_stats["align"] += 1

        except DoubleSuperscriptsError:
            conversion_stats["superscript"] += 1

        except DenominatorNotFoundError:
            conversion_stats["denominator"] += 1

        except Exception as e:
            # Handle other exceptions
            conversion_stats["dunno"] += 1

        # Append equation to big xml
        root.append(mathml_element)
    
    print(f"The latex formulas were converted, here are the stats : {conversion_stats}")

    tree = ET.ElementTree(root)
    tree.write("dataset/equations.xml", encoding="utf-8", xml_declaration=True)
