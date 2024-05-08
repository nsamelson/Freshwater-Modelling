from datasets import load_dataset
from PIL import Image

# Dataset of formulas : https://huggingface.co/datasets/OleehyO/latex-formulas
# There are two datasets: raw_formulas and cleaned_formulas(This dataset has 550K formula-image pairs)
# "We scraped approximately 1 million LaTeX formula image-text pairs from arxiv that were uncleaned and without 
# text segmentation to create the raw_formulas dataset. After cleaning the raw_formulas dataset and integrating 
# it with the im2latex-100K dataset, we obtained the cleaned_formulas dataset, which has 550K formula-image pairs."



# data = load_dataset("OleehyO/latex-formulas", "raw_formulas") 
data = load_dataset("OleehyO/latex-formulas", "cleaned_formulas",trust_remote_code=True) 

example = data["train"][0]
print(example["latex_formula"])
Image.Image.save(example["image"],"example.jpeg")