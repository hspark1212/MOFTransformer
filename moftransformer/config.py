import os
from sacred import Experiment
from moftransformer import __root_dir__
from moftransformer.utils.download import DEFAULT_PRETRAIN_MODEL_PATH

ex = Experiment("pretrained_mof", save_git_info=False)


def _loss_names(d):
    ret = {
        "ggm": 0,  # graph grid matching
        "mpp": 0,  # masked patch prediction
        "mtp": 0,  # mof topology prediction
        "vfp": 0,  # (accessible) void fraction prediction
        "moc": 0,  # metal organic classification
        "bbp": 0,  # building block prediction
        "classification": 0,  # classification
        "regression": 0,  # regression
    }
    ret.update(d)
    return ret


@ex.config
def config():
    """
    # prepare_data
    max_num_atoms = 1000
    min_length = 30
    max_length = 60
    radius = 8
    max_nbr_atoms = 12
    """

    # model
    exp_name = "pretrained_mof"
    seed = 0
    loss_names = _loss_names({"regression":1})

    # graph seeting
    # max_atom_len = 1000  # number of maximum atoms in primitive cell
    atom_fea_len = 64
    nbr_fea_len = 64
    max_graph_len = 300  # number of maximum nodes in graph
    max_nbr_atoms = 12

    # grid setting
    img_size = 30
    patch_size = 5  # length of patch
    in_chans = 1  # channels of grid image
    max_grid_len = -1  # when -1, max_image_len is set to maximum ph*pw of batch images
    draw_false_grid = False

    # transformer setting
    hid_dim = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    mpp_ratio = 0.15

    # downstream
    downstream = ""
    n_classes = 0

    # Optimizer Setting
    optim_type = "adamw"  # adamw, adam, sgd (momentum=0.9)
    learning_rate = 1e-4
    weight_decay = 1e-2
    decay_power = 1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    max_epochs = 100
    max_steps = -1  # num_data * max_epoch // batch_size (accumulate_grad_batches)
    warmup_steps = 0.05  # int or float ( max_steps * warmup_steps)
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # PL Trainer Setting
    resume_from = None
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = os.path.join(__root_dir__, "examples/dataset")
    log_dir = "logs/"
    batch_size = 1024  # desired batch size; for gradient accumulation
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size
    accelerator='gpu'
    devices = 1
    num_nodes = 1

    if os.path.exists(DEFAULT_PRETRAIN_MODEL_PATH):
        load_path = DEFAULT_PRETRAIN_MODEL_PATH
    else:
        load_path = ""

    num_workers = 16  # the number of cpu's core
    precision = 16

    # experiments
    dataset_size = False  # experiments for dataset size with 100 [k] or 500 [k]

    # normalization target
    mean = None
    std = None

    # visualize
    visualize = False  # return attention map