import re
import os

file_path = 'c:/Users/mmbka/OneDrive/AS_DS_prep/technical-study-and-projects/plan/probability_statistics_linearAlgebra_foundations/22_distributions.md'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

def repl(match):
    code_block = match.group(0)
    # Extract the filename from plt.savefig('filename.png'...)
    match_file = re.search(r"plt\.savefig\('([^']+)'", code_block)
    if not match_file:
        return code_block
    
    filename = match_file.group(1)
    abs_path = f"C:/Users/mmbka/OneDrive/AS_DS_prep/technical-study-and-projects/plan/probability_statistics_linearAlgebra_foundations/{filename}"
    
    # Format the caption based on filename
    caption = filename.replace('_', ' ').replace('.png', '').title()
    
    new_text = f"![{caption}]({abs_path})\n\n<details>\n<summary>Python Code for Visualization</summary>\n\n{code_block}\n\n</details>"
    return new_text

# Regex to find python code blocks that contain plt.savefig
new_text = re.sub(r"```python\n(?:(?!```).)*plt\.savefig(?:(?!```).)*\n```", repl, text, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Markdown updated successfully.")
