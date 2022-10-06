from pathlib import Path
import os

class CLICommand:
    """
    run Pathlib

    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('args', nargs='*')

    @staticmethod
    def run(args):
        from moftransformer import __file__ as dir
        run_path = Path(dir).parent/'run.py'
        config = args.args
        print (config)
        os.system(f"python {run_path} with {' '.join(config)}")