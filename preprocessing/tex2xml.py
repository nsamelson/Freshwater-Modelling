from collections import defaultdict
import re
from datasets import load_dataset
from latex2mathml import converter
import xml.etree.ElementTree as ET
from latex2mathml.exceptions import InvalidAlignmentError, DoubleSuperscriptsError, DenominatorNotFoundError
from tqdm import tqdm
from utils import save
from utils import plot


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

def main(debug=False, select_raw=False):
    """
    Download Tex equations, convert to XML and save into XML dataset.
    Dataset of formulas : https://huggingface.co/datasets/OleehyO/latex-formulas
    ### Page info: 
    "We scraped approximately 1 million LaTeX formula image-text pairs from arxiv that were uncleaned and without 
    text segmentation to create the raw_formulas dataset. After cleaning the raw_formulas dataset and integrating 
    it with the im2latex-100K dataset, we obtained the cleaned_formulas dataset, which has 550K formula-image pairs."
    ### Parameters:
    - debug (Bool): Set to True to run in debug mode
    - select_raw (Bool): Set to True to run the full ```raw_formulas``` set
    """

    data = load_dataset("OleehyO/latex-formulas", "cleaned_formulas",trust_remote_code=True) 
    if select_raw:
        data = load_dataset("OleehyO/latex-formulas", "raw_formulas",trust_remote_code=True) 
    conversion_stats = {
        "success":0,
        "unsupported":0,
        "align":0,
        "superscript":0,
        "denominator":0,
        "dunno":0,
    }

    # # Dictionaries to keep track of counts
    # tag_counts = defaultdict(int)
    # text_counts = defaultdict(int)

    # # Regular expression patterns
    # tag_pattern = re.compile(r'</?([a-zA-Z0-9]+)[^>]*>')
    # text_pattern = re.compile(r'>([^<]+)<')

    # Initialize a string to store all XML documents
    root = ET.Element("formulas")

    # Iterate over all equations in the dataset
    for i, formula in enumerate(tqdm(data["train"], desc="Generating XML",unit="equations")):
        latex_formula = formula["latex_formula"]

        if debug:
            if i >=50000:
                break

        # Convert LaTeX formula to MathML
        try:
            # Clean formula
            latex_formula = remove_commands(latex_formula)

            # Convert to MathML
            mathml_element = converter.convert_to_element(latex_formula)
            # mathml_string = ET.tostring(mathml_element, encoding="unicode",method="xml")
            mathml_string = converter.convert(latex_formula)

            if '\\' in mathml_string or '\\\\' in mathml_string:
                conversion_stats["unsupported"] += 1
                continue
            else:
                conversion_stats["success"] += 1

            # # Find all tags
            # tags = tag_pattern.findall(mathml_string)
            # for tag in tags:
            #     if tag in MATHML_TAGS:
            #         tag_counts[tag] += 1
            #     else:
            #         print(tag, latex_formula)
            
            # # Remove tags to get the text content
            # text_segments = tag_pattern.split(mathml_string)
            # for i, segment in enumerate(text_segments):
            #     if i % 2 == 0:  # Even indices contain text content
            #         cleaned_text = re.sub(r'\s+', ' ', segment.strip())  # Clean up whitespace
            #         if cleaned_text:
            #             text_counts[cleaned_text] += 1

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
    # print("Number of different tags: ", len(tag_counts.keys()))
    # print("Number of different labels: ",len(text_counts.keys()))

    # Save dataset
    tree = ET.ElementTree(root)
    tree.write("dataset/equations.xml", encoding="utf-8", xml_declaration=True)

    # # Save stats
    # save.json_dump("out/tags_count.json",tag_counts)
    # save.json_dump("out/labels_count.json",text_counts)
    # plot.plot_labels_frequency()


def remove_commands(text):
    """
    Go through the Tex equation and remove any pattern flagged in ```EXCLUDED_COMMANDS```.
    ### Parameters:
    text (string): input Latex equation
    ### Returns:
    - text (string): output Latex equation
    """
    # Construct a regular expression pattern to match any of the specified commands
    pattern = '|'.join(EXCLUDED_COMMANDS)

    # Replace all occurrences of the pattern with an empty string
    return re.sub(pattern, '', text)
