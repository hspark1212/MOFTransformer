# MOFTransformer version 2.0.0
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
    loss_names = _loss_names({"regression": 1})

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
    decay_power = (
        1  # default polynomial decay, [cosine, constant, constant_with_warmup]
    )
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
    accelerator = "gpu"
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
def ppn_1bar():
    exp_name = "ppn_1bar"
    root_dataset = (
        "/usr/data/transfer_learning/downstream_public/0_insilico_ppn/dataset"
    )
    downstream = "1bar"
    max_epochs = 20
    batch_size = 32
    mean = 3.79
    std = 5.32


@ex.named_config
def ppn_65bar():
    exp_name = "ppn_65bar"
    root_dataset = (
        "/usr/data/transfer_learning/downstream_public/0_insilico_ppn/dataset"
    )
    downstream = "65bar"
    max_epochs = 20
    batch_size = 32
    mean = 117.78
    std = 30.75


"""
in silico COF
"""


@ex.named_config
def cof_lowbar():
    exp_name = "cof_lowbar"
    root_dataset = (
        "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    )
    downstream = "lowbar"
    max_epochs = 20
    batch_size = 32
    mean = 23.750
    std = 17.166


@ex.named_config
def cof_highbar():
    exp_name = "cof_highbar"
    root_dataset = (
        "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    )
    downstream = "highbar"
    max_epochs = 20
    batch_size = 32
    mean = 159.076
    std = 38.164


@ex.named_config
def cof_logkh():
    exp_name = "cof_logkh"
    root_dataset = (
        "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    )
    downstream = "logkh"
    max_epochs = 20
    batch_size = 32
    mean = -10.975
    std = 0.563


@ex.named_config
def cof_qst():
    exp_name = "cof_qst"
    root_dataset = (
        "/usr/data/transfer_learning/downstream_public/1_insilico_cof/dataset"
    )
    downstream = "qst"
    max_epochs = 20
    batch_size = 32
    mean = -14.793
    std = 4.542


"""
pcod zeolite
"""


@ex.named_config
def zeo_qst():
    exp_name = "zeo_qst"
    root_dataset = (
        "/usr/data/transfer_learning/downstream_public/2_pcod_zeolite/dataset"
    )
    downstream = "qst"
    max_epochs = 20
    batch_size = 32
    mean = 19.052
    std = 3.169


@ex.named_config
def zeo_unitlesskh():
    exp_name = "zeo_unitlesskh"
    root_dataset = (
        "/usr/data/transfer_learning/downstream_public/2_pcod_zeolite/dataset"
    )
    downstream = "unitlesskh"
    max_epochs = 20
    batch_size = 32
    mean = 19.725
    std = 12.317


"""
mof downstream
"""


@ex.named_config
def mof_raspa_100bar():
    exp_name = "mof_raspa_100bar"
    # root_dataset = ##
    downstream = "raspa_100bar"
    max_epochs = 20
    batch_size = 32
    mean = 487.866
    std = 63.504


@ex.named_config
def mof_diffusivity_log():
    exp_name = "mof_diffusivity_log"
    # root_dataset = ##
    downstream = "diffusivity_log"
    max_epochs = 20
    batch_size = 32
    mean = -8.306
    std = 1.490


@ex.named_config
def mof_bandgap():
    exp_name = "mof_bandgap"
    # root_dataset = ##
    downstream = "bandgap"
    max_epochs = 20
    batch_size = 32
    mean = 2.086
    std = 1.131


@ex.named_config
def mof_n2uptake():
    exp_name = "mof_n2uptake"
    # root_dataset = ##
    downstream = "n2uptake"
    max_epochs = 20
    batch_size = 32
    mean = 0.3999
    std = 0.337


@ex.named_config
def mof_o2uptake():
    exp_name = "mof_o2uptake"
    # root_dataset = ##
    downstream = "o2uptake"
    max_epochs = 20
    batch_size = 32
    mean = 0.387
    std = 0.241


@ex.named_config
def mof_n2diffusivity_dilute():
    exp_name = "mof_n2diffusivity_dilute"
    # root_dataset = ##
    downstream = "n2diffusivity_dilute"
    max_epochs = 20
    batch_size = 32
    mean = 0.000187
    std = 0.000176


@ex.named_config
def mof_o2diffusivity_dilute():
    exp_name = "mof_o2diffusivity_dilute"
    # root_dataset = ##
    downstream = "o2diffusivity_dilute"
    max_epochs = 20
    batch_size = 32
    mean = 0.000185
    std = 0.000162


@ex.named_config
def mof_henry_co2():
    exp_name = "mof_henry_co2"
    # root_dataset = ##
    downstream = "henry_co2"
    max_epochs = 20
    batch_size = 32
    mean = -3.554
    std = 1.120


@ex.named_config
def mof_tsr():
    # thermal stability regression
    exp_name = "mof_tsr"
    # root_dataset = ##
    downstream = "tsr"
    max_epochs = 20
    batch_size = 32
    mean = 361.322
    std = 88.122


@ex.named_config
def mof_ssc():
    # solvent stability classification
    exp_name = "mof_ssc"
    # root_dataset = ##
    downstream = "ssc"
    max_epochs = 20
    batch_size = 32
    mean = 0.592
    std = 0.491
    loss_names = _loss_names({"classification": 1})
    n_classes = 2


"""
downstream example (H2 uptake)
"""


@ex.named_config
def mof_h2_uptake():
    exp_name = "mof_h2_uptake"
    # root_dataset = ##
    downstream = "h2_uptake"
    max_epochs = 20
    batch_size = 32
    mean = 488.029
    std = 62.690


@ex.named_config
def cof_h2_uptake():
    exp_name = "cof_h2_uptake"
    # root_dataset = ##
    downstream = "h2_uptake"
    max_epochs = 20
    batch_size = 32
    mean = 485.978
    std = 80.930


@ex.named_config
def ppn_h2_uptake():
    exp_name = "ppn_h2_uptake"
    # root_dataset = ##
    downstream = "h2_uptake"
    max_epochs = 20
    batch_size = 32
    mean = 465.196
    std = 117.529


@ex.named_config
def zeo_h2_uptake():
    exp_name = "zeo_h2_uptake"
    # root_dataset = ##
    downstream = "h2_uptake"
    max_epochs = 20
    batch_size = 32
    mean = 259.878
    std = 112.928


"""
downstream example (H2 working capacity)
"""


@ex.named_config
def mof_h2_wc():
    exp_name = "mof_h2_wc"
    # root_dataset = ##
    downstream = "h2_wc"
    max_epochs = 20
    batch_size = 32
    mean = 320.019
    std = 87.993


@ex.named_config
def cof_h2_wc():
    exp_name = "cof_h2_wc"
    # root_dataset = ##
    downstream = "h2_wc"
    max_epochs = 20
    batch_size = 32
    mean = 326.740
    std = 84.470


@ex.named_config
def ppn_h2_wc():
    exp_name = "ppn_h2_wc"
    # root_dataset = ##
    downstream = "h2_wc"
    max_epochs = 20
    batch_size = 32
    mean = 300.407
    std = 116.746


@ex.named_config
def zeo_h2_wc():
    exp_name = "zeo_h2_wc"
    # root_dataset = ##
    downstream = "h2_wc"
    max_epochs = 20
    batch_size = 32
    mean = 28.667
    std = 25.660


@ex.named_config
def total_h2_wc():
    exp_name = "total_h2_wc"
    # root_dataset = ##
    downstream = "h2_wc"
    max_epochs = 20
    batch_size = 32
    mean = 241.300
    std = 152.052


"""
curated COF few shot
"""


@ex.named_config
def ccof_bandgap():
    exp_name = "ccof_bandgap"
    # root_dataset = ##
    downstream = "ccof_bandgap"
    max_epochs = 20
    batch_size = 32
    mean = 1.619
    std = 0.557


@ex.named_config
def qmof_bandgap():
    exp_name = "qmof_bandgap"
    # root_dataset = ##
    downstream = "qmof_bandgap"
    max_epochs = 20
    batch_size = 32
    mean = 2.086
    std = 1.131
