class CLICommand:
    """
    Download pre-trained model, database, and fine-tuned model
    """

    @staticmethod
    def add_arguments(parser):
        from moftransformer.utils.download import DEFAULT_PRETRAIN_MODEL_PATH, DEFAULT_COREMOF_PATH, DEFAULT_QMOF_PATH
        add = parser.add_argument
        add ('target', nargs='+', help='(str) pretrain_model, finetuned_model, coremof, qmof, or hmof\n'
                                       'pre-trained model (MTP/MOC/VFP), fine-tuned model (h2 uptake and band gap),\n'
                                       'and database which contain graph-data and grid-data for "coremof", "qmof", and "hmof"\n')
        add ('--outdir', '-o', help=f'The Path where downloaded data will be stored. \n'
                                    f'default : (pretrain_model) {DEFAULT_PRETRAIN_MODEL_PATH} \n'
                                    f'          (coremof) {DEFAULT_COREMOF_PATH}/\n'
                                    f'          (qmof) {DEFAULT_QMOF_PATH}/\n'
                                    f'          (hmof) {DEFAULT_QMOF_PATH}/hmof/\n'
                                    f'          (finetuned_model) [path_moftransformer]/example/finetuned_model/', default=None)
        add ('--remove_tarfile', '-r', action='store_true', help='remove tar.gz file for download database (coremof, qmof, and hmof)')

    @staticmethod
    def run(args):
        from moftransformer.utils.download import (
            download_pretrain_model,
            download_coremof,
            download_finetuned_model,
            download_hmof,
            download_qmof,
        )

        func_dic = {'pretrain_model':download_pretrain_model,
                    'coremof':download_coremof,
                    'hmof':download_hmof,
                    'qmof':download_qmof,
                    'finetuned_model':download_finetuned_model}

        for stuff in args.target:
            if stuff not in func_dic.keys():
                raise ValueError(f'target must be {", ".join(func_dic.keys())}, not {stuff}')

        for stuff in args.target:
            func = func_dic[stuff]
            if func.__code__.co_argcount == 1:
                func(args.outdir)
            else:
                func(args.outdir, args.remove_tarfile)
