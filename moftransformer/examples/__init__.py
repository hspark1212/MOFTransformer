import os
from moftransformer import __root_dir__


example_path = {'data_root':os.path.join(__root_dir__, 'examples/dataset'),
                'downstream':'example'}
raw_cif_path = os.path.join(__root_dir__, 'examples/raw')
visualize_example_path = os.path.join(__root_dir__, 'examples/visualize/dataset')
