from collections import defaultdict
import re
from datasets import load_dataset
from latex2mathml import converter
import xml.etree.ElementTree as ET
from latex2mathml.exceptions import InvalidAlignmentError, DoubleSuperscriptsError, DenominatorNotFoundError
from tqdm import tqdm
from utils import save
from utils import plot
import subprocess
import os
import logging
import json

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


def main(debug=False, select_raw=False,batch_size=1000):
    """
    Download Tex equations, convert to XML and save into XML dataset.
    Dataset of formulas : https://huggingface.co/datasets/OleehyO/latex-formulas
    
    Page info: 
    "We scraped approximately 1 million LaTeX formula image-text pairs from arxiv that were uncleaned and without 
    text segmentation to create the raw_formulas dataset. After cleaning the raw_formulas dataset and integrating 
    it with the im2latex-100K dataset, we obtained the cleaned_formulas dataset, which has 550K formula-image pairs."
    
    Args:
    - debug (Bool): Set to True to run in debug mode
    - select_raw (Bool): Set to True to run the full ```raw_formulas``` set
    """

    # with open("dataset/latex_examples.json","r") as f:
    #     data = json.load(f)

    data = load_dataset("OleehyO/latex-formulas", "cleaned_formulas",trust_remote_code=True) 
    if select_raw:
        data = load_dataset("OleehyO/latex-formulas", "raw_formulas",trust_remote_code=True) 
    conversion_stats = {
        "success":0,
        "error": 0,
        "TypeError":0,
        "NoneType":0
        # "unsupported":0,
        # "align":0,
        # "superscript":0,
        # "denominator":0,
        # "dunno":0,
    }

    # Create the root element with the <span class="katex"> tag
    ET.register_namespace('', 'http://www.w3.org/1998/Math/MathML')
    root = ET.Element("span", attrib={"class": "katex"})

    all_equations = [formula["latex_formula"] for formula in data["train"]]

    # Iterate over all equations in the dataset
    for i,batch in enumerate(tqdm(process_equations_in_batches(all_equations, batch_size), desc="Processing batches",unit="batch")):

        if debug:
            if i >=5:
                break
        
        # Clean formula
        cleaned_batch = [remove_commands(formula) for formula in batch]

        # Convert to mathml
        mathml_results = call_js(cleaned_batch)
        
        if mathml_results:
            for mathml_string in mathml_results:
                try:
                    # Parse the MathML string into an ElementTree element
                    span_element = ET.fromstring(mathml_string)
                    mathml_element = span_element.find("{http://www.w3.org/1998/Math/MathML}math")

                    if mathml_element is not None:
                        # Append the <math> element to the root <span class="katex"> element
                        root.append(mathml_element)
                        conversion_stats["success"] += 1
                    else:
                        conversion_stats["NoneType"] += 1
                except TypeError as e:
                    # print(e, ": ",latex_formula, mathml_string)
                    conversion_stats["TypeError"] += 1
                except Exception as e:
                    conversion_stats["error"] += 1
        

        def mathmlConverter(latex_formula):
            # Convert LaTeX formula to MathML
            try:
                # Clean formula
                latex_formula = remove_commands(latex_formula)

                # Convert to MathML
                mathml_element = converter.convert_to_element(latex_formula)
                # mathml_string = ET.tostring(mathml_element, encoding="unicode",method="xml")
                mathml_string = converter.convert(latex_formula)
                if "1mm" in mathml_string:
                    print(latex_formula, mathml_string)

                if '\\' in mathml_string or '\\\\' in mathml_string:
                    conversion_stats["unsupported"] += 1
                    # continue
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
    
    
    print(f"The latex formulas were converted, here are the stats : {conversion_stats}")

    # Save dataset
    tree = ET.ElementTree(root)
    tree.write("dataset/equations.xml", encoding="utf-8", xml_declaration=True)

def process_equations_in_batches(equations, batch_size=1000):
    for i in range(0, len(equations), batch_size):
        yield equations[i:i + batch_size]

def remove_commands(text):
    """
    Go through the Tex equation and remove any pattern flagged in ```EXCLUDED_COMMANDS```.
    
    Args:
    - text (string): input Latex equation
    
    Returns:
    - text (string): output Latex equation
    """
    # Construct a regular expression pattern to match any of the specified commands
    pattern = '|'.join(EXCLUDED_COMMANDS)

    # Replace all occurrences of the pattern with an empty string
    return re.sub(pattern, '', text)

def call_js(latex_equations, paper_id=""):
    try:
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.dirname(current_file_path)
        script_path = os.path.join(root_folder,"node" ,"tex2mathml_simple.js")

        # Add the directory where node is installed to PATH
        env = os.environ.copy()
        node_bin_dir = "/data/nsam947/libs/node-v20.13.1-linux-x64/bin"
        env["PATH"] = node_bin_dir + os.pathsep + env["PATH"]

        # logging.debug("Running tex2mathml.js with environment PATH: {}".format(env["PATH"]))

        result = subprocess.run(
            [script_path],
            input=json.dumps(latex_equations),
            cwd=root_folder,
            env=env,
            universal_newlines=True,
            text=True,
            capture_output=True,
            timeout=120
        )

        # logging.debug("stderr output: {}".format(result.stderr))
        # logging.debug("stdout output: {}".format(result.stdout))

        raw_output = result.stdout.strip()
        # logging.debug("Raw output before JSON parsing: {}".format(raw_output))

        # if result.stderr:
        #     logging.warning("Unexpected error in tex2mathml.js (Arxiv ID: {}):".format(paper_id) + result.stderr)

        if raw_output:
            try:
                result_data = json.loads(raw_output)
                return result_data
            except json.JSONDecodeError as e:
                # logging.error("JSON decoding failed: {}".format(e))
                return None

    except subprocess.TimeoutExpired:
        # logging.warning("Timeout for paper {}: \n".format(paper_id) + "\n")
        return False
    except Exception as e:
        # logging.error("Error calling tex2mathml.js: {}".format(e))
        return False
