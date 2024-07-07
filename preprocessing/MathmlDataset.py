import html
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
from config import ROOT_DIR

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

DATASET_NAMES = {
    "OleehyO": "OleehyO/latex-formulas",
    "sample": "dataset/latex_examples.json",
    "Pfahler": "arxiv/preprocessed",
}

class MathmlDataset(Dataset):
    def __init__(self, xml_name:str, latex_set="OleehyO", force_reload=False, debug=False, select_raw=False, batch_size=1000):
        self.latex_set = latex_set if latex_set in DATASET_NAMES.keys() else None
        self.force_reload = force_reload
        self.debug = debug
        self.select_raw = select_raw
        self.batch_size = batch_size
        self.root = None
        self.stats = {
            "success": 0,
            "error": 0,
            "TypeError": 0,
            "NoneType": 0
        }

        if self.latex_set is None:
            raise Exception("No Latex set found")
        
        root_path = os.path.join(ROOT_DIR,"data/pre_processed")
        
        self.xml_dir = os.path.join(root_path,xml_name)
        self.xml_path = os.path.join(root_path,xml_name,"raw/equations.xml")
        self.latex_path = DATASET_NAMES.get(latex_set,None)

        if not os.path.exists(self.xml_dir):
            os.makedirs(self.xml_dir)

        raw_dir = os.path.join(self.xml_dir,"raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        if not os.path.exists(self.xml_path) or force_reload:
            self.process_data()
        else:
            self.load_xml()

    
    def process_data(self):

        print("Loading Latex dataset...")
        all_equations = self.load_latex()

        # Create the root element with the <span class="katex"> tag
        ET.register_namespace('', 'http://www.w3.org/1998/Math/MathML')
        self.root = ET.Element("span", attrib={"class": "katex"})

        # Go through by batch and convert to XML
        for i,batch in enumerate(tqdm(process_equations_in_batches(all_equations,self.batch_size), desc="Generating XML",unit=" batch", total=int(len(all_equations)/self.batch_size))):
            if self.debug and i>5:
                break

            # Clean formulas and convert with katex nodejs
            cleaned_batch = [remove_commands(formula) for formula in batch]
            mathml_results = call_js(cleaned_batch)

            if mathml_results:
                for mathml_string in mathml_results:
                    try:
                        # Parse the MathML string into an ElementTree element
                        span_element = ET.fromstring(mathml_string)
                        mathml_element = span_element.find("{http://www.w3.org/1998/Math/MathML}math")

                        if mathml_element is not None:
                            # Append the <math> element to the root <span class="katex"> element
                            self.root.append(mathml_element)
                            self.stats["success"] += 1
                        else:
                            self.stats["NoneType"] += 1
                    except TypeError as e:
                        self.stats["TypeError"] += 1
                    except Exception as e:
                        self.stats["error"] += 1
        # Save dataset
        print("Saving XML...")
        tree = ET.ElementTree(self.root)
        tree.write(self.xml_path, encoding="utf-8", xml_declaration=True)

    

    def load_latex(self):
        """
        Method to load the selected latex set
        """
        if self.latex_set == "OleehyO":
            formulas = "raw_formulas" if self.select_raw else "cleaned_formulas"
            data = load_dataset(self.latex_path, formulas, trust_remote_code=True, split="train")
            return data["latex_formula"]
        
        elif self.latex_set == "sample":
            with open(self.latex_path,"r") as f:
                data = json.load(f)
            return data["train"]

        elif self.latex_set == "Pfahler":
            # TODO: implement pfahler method
            return []


    def load_xml(self):
        tree = ET.parse(self.xml_path)
        self.root = tree.getroot()

    def __len__(self):
        return len(self.root)

    def __getitem__(self, idx):
        return self.root[idx]



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
    

if __name__=="__main__":
    # Usage example
    print("starting stuff")
    dataset = MathmlDataset("default", force_reload=False, debug=False)
    print(f"The latex formulas were converted, here are the stats : {dataset.stats}")
    print(len(dataset))
    print(ET.tostring(dataset[0], encoding='unicode'))