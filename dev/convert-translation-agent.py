#!/usr/bin/env python

from nbconvert.exporters.script import ScriptExporter
from nbconvert.preprocessors import Preprocessor
import nbformat

class FilterCellsPreprocessor(Preprocessor):
    def preprocess(self, nb, resources):
        print(f"Number of notebook cells found: {len(nb.cells)}")

        nb.cells = [cell for cell in nb.cells if 'export' in cell.metadata.get('tags', [])]

        print(f"Number of notebook cells to keep: {len(nb.cells)}")

        return nb, resources

filter_cells_preprocessor = FilterCellsPreprocessor()

# Load notebook
with open("agentic-translation.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

# Create a filter to remove the notebook cells that don't have the 'export' tag in metadata
nb, resources = filter_cells_preprocessor.preprocess(nb, None)

# Export the notebook while filtering cells
exporter = ScriptExporter()
script, resources = exporter.from_notebook_node(nb)

# Save the script to a .py file
with open("../translation_agent/translation_agent_util.py", "w") as f:
    f.write(script)

print("Done")
