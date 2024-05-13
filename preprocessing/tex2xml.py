import re
from datasets import load_dataset
from latex2mathml import converter
import xml.etree.ElementTree as ET
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
from latex2mathml.exceptions import InvalidAlignmentError, DoubleSuperscriptsError, DenominatorNotFoundError
from tqdm import tqdm

EXCLUDED_COMMANDS = [
    r"\\begin{align\*?}", r"\\end{align\*?}",  # align and align*
    r"\\begin{equation\*?}", r"\\end{equation\*?}",  # equation and equation*
    r"\\mbox\{[^\}]*\}",  # \mbox{...}
    r"\\label\{[^\}]*\}",  # \label{...}
    r"\\def\b",  # \def
    r"\\text\b",  # \text
    r"\\sbox\b",  # \sbox
    r"\\nonumber\b",  # \nonumber
    r"\\notag\b",  # \notag
    r"\\value\b",  # \value
    r"\\todo\b",  # \todo
    r"\\def\b",  # \def
    r"\\scalebox\b",  # \scalebox
    r"\\vspace\b",  # \vspace
    r"\\ensuremath\b",  # \ensuremath
    r"\\hfill\b",  # \hfill
    r"\\footnote\b",  # \footnote
    r"\\footnotemark\b",  # \footnotemark
    r"\\marginpar\b",  # \marginpar
    r"\\xspace\b",  # \xspace
    r"\\norm\b",  # \norm
    r"\\lefteqn\b",  # \lefteqn
    r"\\textsc\b",  # \textsc
    r"\\newtheorem\b",  # \newtheorem
    r"\\par\b",  # \par
    r"\\vskip\b",  # \vskip
    r"\\baselineskip\b",  # \baselineskip
    r"\\textsuperscript\b",  # \textsuperscript
    r"\\title\b",  # \title
    r"\\author\b",  # \author
    r"\\makeatother\b",  # \makeatother
    r"\\mathbb\b"  # \mathbb
]

def main(debug=True):
    """
    Download Tex equations, convert to XML and save into XML dataset.
    Dataset of formulas : https://huggingface.co/datasets/OleehyO/latex-formulas
    Page info: "We scraped approximately 1 million LaTeX formula image-text pairs from arxiv that were uncleaned and without 
    text segmentation to create the raw_formulas dataset. After cleaning the raw_formulas dataset and integrating 
    it with the im2latex-100K dataset, we obtained the cleaned_formulas dataset, which has 550K formula-image pairs."
    """
    # data = load_dataset("OleehyO/latex-formulas", "raw_formulas",trust_remote_code=True) 
    data = load_dataset("OleehyO/latex-formulas", "cleaned_formulas",trust_remote_code=True) 

    conversion_stats = {
        "success":0,
        "unsupported":0,
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

        if debug:
            if i >=100:
                break

        # Convert LaTeX formula to MathML
        try:
            # Clean formula
            latex_formula = remove_commands(latex_formula)
            # print(latex_formula)

            # Convert to MathML
            mathml_element = converter.convert_to_element(latex_formula)
            mathml_string = ET.tostring(mathml_element, encoding="unicode",method="xml")



            if '\\' in mathml_string or '\\\\' in mathml_string:
                conversion_stats["unsupported"] += 1
                continue
            else:
                conversion_stats["success"] += 1

            # Append equation to big xml
            root.append(mathml_element)

        except InvalidAlignmentError:
            conversion_stats["align"] += 1

        except DoubleSuperscriptsError:
            conversion_stats["superscript"] += 1

        except DenominatorNotFoundError:
            conversion_stats["denominator"] += 1

        except Exception as e:
            # Handle other exceptions
            conversion_stats["dunno"] += 1
            # print(e)

    
    print(f"The latex formulas were converted, here are the stats : {conversion_stats}")

    tree = ET.ElementTree(root)
    tree.write("dataset/equations.xml", encoding="utf-8", xml_declaration=True)




def remove_commands(text):

    # Construct a regular expression pattern to match any of the specified commands
    pattern = '|'.join(EXCLUDED_COMMANDS)

    # Replace all occurrences of the pattern with an empty string
    return re.sub(pattern, '', text)
