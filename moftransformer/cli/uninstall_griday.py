class CLICommand:
    """
    Install package <GIRDAY> which calculated energy-grid.

    """

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        from moftransformer.utils import uninstall_griday
        uninstall_griday()
