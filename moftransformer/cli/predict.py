# MOFTransformer version 2.1.1
from itertools import chain
from moftransformer.predict import predict

str_kwargs_names = {
    'loss_names': "One or more of the following loss : 'regression', 'classification', 'mpt', 'moc', and 'vfp'"
}

int_kwargs_names = {
    'n_classes': "Number of classes when your loss is 'classification'",
    'batch_size': 'desired batch size; for gradient accumulation',
    'per_gpu_batchsize': 'you should define this manually with per_gpu_batch_size',
    'num_nodes': 'Number of GPU nodes for distributed training.',
    'num_workers': "the number of cpu's core",
    'seed': 'The random seed for pytorch_lightning.',
    'max_steps': 'num_data * max_epoch // batch_size (accumulate_grad_batches). If -1, set max_steps automatically.',
}

float_kwargs_names = {
    'mean': 'mean for normalizer. If None, it is automatically obtained from the train dataset.',
    'std': 'standard deviation for normalizer. If None, it is automatically obtained from the train dataset.',
}


none_kwargs_names = {
    'accelerator' : 'Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto") '
    'as well as custom accelerator instances.',
    'devices' : 'Number of devices to train on (int), which devices to train on (list or str), or "auto". '
    'It will be mapped to either gpus, tpu_cores, num_processes or ipus, based on the accelerator type ("cpu", "gpu", "tpu", "ipu", "auto").',
}


class CLICommand:
    """
    run moftransformer code

    ex) moftransformer run downstream=example downstream=bandgap devices=1 max_epochs=10

    """

    @staticmethod
    def add_arguments(parser):
        #parser.add_argument("args", nargs="*")
        parser.add_argument("--root_dataset", "-r", type=str, 
                            help="A folder containing graph data, grid data, and json of MOFs that you want to train or test. "\
                            "The way to make root_dataset is at this link (https://hspark1212.github.io/MOFTransformer/dataset.html)"
                            )
        parser.add_argument('--load_path', '-l', type=str, help='Path for model you want to load and predict (*.ckpt).')
        parser.add_argument('--downstream', "-d", type=str, default=None, help="Name of user-specific task (e.g. bandgap, gasuptake, etc). "
                            "if downstream is None, target json is 'train.json', 'val.json', and 'test.json'",
                            )
        parser.add_argument('--split', "-s", default='all', type=str, help="(optional) The split you want to predict on your dataset ('all', 'train', 'test', or 'val')")
        parser.add_argument('--save_dir', "-sd", type=str, default=None,
                            help='(optional) Path for directory you want to save *.csv file. (default : None -> path for loaded model)'
                            )

        for key, value in str_kwargs_names.items():
            parser.add_argument(f"--{key}", type=str, required=False, help=f"(optional) {value}")

        for key, value in none_kwargs_names.items():
            parser.add_argument(f"--{key}", required=False, help=f"(optional) {value}")

        for key, value in int_kwargs_names.items():
            parser.add_argument(f"--{key}", type=int, required=False, help=f"(optional) {value}")

        for key, value in float_kwargs_names.items():
            parser.add_argument(f"--{key}", type=float, required=False, help=f"(optional) {value}")

    @staticmethod
    def run(args):
        from moftransformer import __root_dir__
        
        root_dataset = args.root_dataset
        load_path = args.load_path
        downstream = args.downstream
        split = args.split
        save_dir = args.save_dir

        kwargs = {}
        for key in chain(str_kwargs_names.keys(), 
                         none_kwargs_names.keys(),
                         int_kwargs_names.keys(),
                         float_kwargs_names.keys(),
                         ):
            if value := getattr(args, key):
                kwargs[key] = value
                
        predict(
            root_dataset,
            load_path,
            downstream=downstream,
            split=split,
            save_dir=save_dir,
            **kwargs
        )
