# MOFTransformer version 2.0.0
import os
from moftransformer import __root_dir__


example_path = {
    "root_cif": os.path.join(__root_dir__, "examples/raw"),
    "root_dataset": os.path.join(__root_dir__, "examples/dataset"),
    "data_root": os.path.join(__root_dir__, "examples/dataset"),
    "downstream": "example",
}
raw_cif_path = os.path.join(__root_dir__, "examples/raw")
visualize_example_path = os.path.join(__root_dir__, "examples/visualize/dataset")
