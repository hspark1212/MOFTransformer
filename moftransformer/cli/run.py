from pathlib import Path
import os

class CLICommand:
    """
    run moftransformer code

    ex) moftransformer run downstream='example' num_gpus=1 max_epochs=10

    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('args', nargs='*')

    @staticmethod
    def run(args):
        from moftransformer import __root_dir__
        run_path = Path(__root_dir__)/'run.py'
        config = args.args
        print (config)
        os.system(f"python {run_path} with {' '.join(config)}")