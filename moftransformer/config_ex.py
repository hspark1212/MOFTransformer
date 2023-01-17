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
        "bbc": 0,  # building block classification
        "classification": 0,  # classification
        "regression": 0,  # regression
    }
    ret.update(d)
    return ret


@ex.config
def config():
    """
    # prepare_data
    max_num_atoms = 300
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
    root_dataset = os.path.join(__root_dir__, "examples/dataset")
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

    # normalization target
    mean = None
    std = None

    # visualize
    visualize = False  # return attention map


@ex.named_config
def example():
    exp_name = "example"
    root_dataset = "moftransformer/examples/dataset"
    downstream = "example"
    max_epochs = 20
    batch_size = 32

"""
pretraining
"""

@ex.named_config
def mtp_bbc_vfp():
    load_path = ""
    exp_name = "mtp_bbc_vfp"
    root_dataset = "/usr/data/transfer_learning/dataset"
    loss_names = _loss_names({"mtp": 1, "bbc": 1, "vfp": 1})
    per_gpu_batchsize = 4

"""
fine-tuning (transfer learining)
"""
@ex.named_config
def ppn_1bar_scratch():
    load_path = ""
    exp_name = "ppn_1bar_scratch"
    root_dataset = "/usr/data/transfer_learning/downstream_public/0_insilico_ppn/dataset"
    downstream = "1bar"
    max_epochs = 20
    batch_size = 32
    mean = 3.79
    std = 5.32

@ex.named_config
def ppn_1bar():
    exp_name = "ppn_1bar"
    root_dataset = "/usr/data/transfer_learning/downstream_public/0_insilico_ppn/dataset"
    downstream = "1bar"
    max_epochs = 20
    batch_size = 32
    mean = 3.79
    std = 5.32

@ex.named_config
def ppn_65bar_scratch():
    load_path = ""
    exp_name = "ppn_65bar_scratch"
    root_dataset = "/usr/data/transfer_learning/downstream_public/0_insilico_ppn/dataset"
    downstream = "65bar"
    max_epochs = 20
    batch_size = 32
    mean = 117.78
    std = 30.75


@ex.named_config
def ppn_65bar():
    exp_name = "ppn_65bar"
    root_dataset = "/usr/data/transfer_learning/downstream_public/0_insilico_ppn/dataset"
    downstream = "65bar"
    max_epochs = 20
    batch_size = 32
    mean = 117.78
    std = 30.75

"""
in silico COF
"""
@ex.named_config
def cof_lowbar_scratch():
    load_path = ""
    exp_name = "cof_lowbar_scratch"
    root_dataset = "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    downstream = "lowbar"
    max_epochs = 20
    batch_size = 32
    mean = 23.750
    std = 17.166

@ex.named_config
def cof_lowbar():
    exp_name = "cof_lowbar"
    root_dataset = "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    downstream = "lowbar"
    max_epochs = 20
    batch_size = 32
    mean = 23.750
    std = 17.166

@ex.named_config
def cof_highbar_scratch():
    load_path = ""
    exp_name = "cof_highbar_scratch"
    root_dataset = "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    downstream = "highbar"
    max_epochs = 20
    batch_size = 32
    mean = 159.076
    std = 38.164

@ex.named_config
def cof_highbar():
    exp_name = "cof_highbar"
    root_dataset = "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    downstream = "highbar"
    max_epochs = 20
    batch_size = 32
    mean = 159.076
    std = 38.164

@ex.named_config
def cof_logkh_scratch():
    load_path = ""
    exp_name = "cof_logkh_scratch"
    root_dataset = "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    downstream = "logkh"
    max_epochs = 20
    batch_size = 32
    mean = -10.975
    std = 0.563

@ex.named_config
def cof_logkh():
    exp_name = "cof_logkh"
    root_dataset = "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    downstream = "logkh"
    max_epochs = 20
    batch_size = 32
    mean = -10.975
    std = 0.563

@ex.named_config
def cof_qst_scratch():
    load_path = ""
    exp_name = "cof_qst_scratch"
    root_dataset = "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    downstream = "qst"
    max_epochs = 20
    batch_size = 32
    mean = -14.793
    std = 4.542

@ex.named_config
def cof_qst():
    exp_name = "cof_qst"
    root_dataset = "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    downstream = "qst"
    max_epochs = 20
    batch_size = 32
    mean = -14.793
    std = 4.542

"""
pcod zeolite
"""

@ex.named_config
def zeo_qst_scratch():
    load_path = ""
    exp_name = "zeo_qst_scratch"
    root_dataset = "/usr/data/transfer_learning/downstream_public/2_pcod_zeolite/dataset"
    downstream = "qst"
    max_epochs = 20
    batch_size = 32
    mean = 19.052
    std = 3.169

@ex.named_config
def zeo_qst():
    exp_name = "zeo_qst"
    root_dataset = "/usr/data/transfer_learning/downstream_public/2_pcod_zeolite/dataset"
    downstream = "qst"
    max_epochs = 20
    batch_size = 32
    mean = 19.052
    std = 3.169

@ex.named_config
def zeo_unitlesskh_scratch():
    load_path = ""
    exp_name = "zeo_unitlesskh_scratch"
    root_dataset = "/usr/data/transfer_learning/downstream_public/2_pcod_zeolite/dataset"
    downstream = "unitlesskh"
    max_epochs = 20
    batch_size = 32
    mean = 19.725
    std = 12.317

@ex.named_config
def zeo_unitlesskh():
    exp_name = "zeo_unitlesskh"
    root_dataset = "/usr/data/transfer_learning/downstream_public/2_pcod_zeolite/dataset"
    downstream = "unitlesskh"
    max_epochs = 20
    batch_size = 32
    mean = 19.725
    std = 12.317