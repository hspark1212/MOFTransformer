from sacred import Experiment

ex = Experiment("pretrained_mof")


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
    use_transformer = True  # use graph embedding + vision transformer 3D
    loss_names = _loss_names({})

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
    data_root = ""
    log_dir = "result"
    batch_size = 1024  # desired batch size; for gradient accumulation
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size
    num_gpus = 2
    num_nodes = 1
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

@ex.named_config
def downstream_example():
    exp_name = "downstream_example"
    data_root = "examples/dataset"
    log_dir = "examples/logs"
    downstream = "example"
    load_path = "examples/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = None
    std = None



@ex.named_config
def env_neuron():
    # data_root = "/scratch/x2287a03/ver4"
    pass


@ex.named_config
def small_transformer():
    # model
    hid_dim = 512
    num_heads = 8
    num_layers = 4


@ex.named_config
def medium_transformer():
    # model
    hid_dim = 512
    num_heads = 8
    num_layers = 8



"""
before release
"""


@ex.named_config
def downstream_topology():
    exp_name = "downstream_topology_20k"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "topology"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"classification": 1})
    n_classes = 1100


@ex.named_config
def downstream_1bar():
    exp_name = "downstream_1bar_20k"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "1bar"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 66.960
    std = 51.656


@ex.named_config
def downstream_100bar():
    exp_name = "downstream_100bar_20k"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "100bar"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 447.920
    std = 68.100

@ex.named_config
def downstream_raspa_100bar():
    exp_name = "downstream_raspa_100bar_20k"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "raspa_100bar"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 487.841
    std = 63.088


@ex.named_config
def downstream_5_scaled():
    exp_name = "downstream_5_scaled_20k"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "5_scaled"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})


@ex.named_config
def downstream_100_scaled():
    exp_name = "downstream_100_scaled_20k"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "100_scaled"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})


@ex.named_config
def downstream_uptake():
    exp_name = "downstream_uptake"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "uptake"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    mean = 291.453
    std = 95.503


@ex.named_config
def downstream_bandgap():
    exp_name = "downstream_bandgap"
    data_root = "/home/data/pretrained_mof/qmof/dataset/20k"
    log_dir = "result_downstream"
    downstream = "bandgap"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    mean = 2.097
    std = 1.088

@ex.named_config
def downstream_diffusivity():
    exp_name = "downstream_diffusivity"
    data_root = "/home/data/pretrained_mof/qmof/dataset/20k"
    log_dir = "result_downstream"
    downstream = "diffusivity"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    mean = 0.000506
    std = 0.000711


@ex.named_config
def downstream_diffusivity_log():
    exp_name = "downstream_diffusivity_log"
    data_root = "/home/data/pretrained_mof/qmof/dataset/20k"
    log_dir = "result_downstream"
    downstream = "diffusivity_log"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    mean = -8.300
    std = 1.484




@ex.named_config
def downstream_bulkmodulus():
    exp_name = "downstream_bulkmodulus"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "bulkmodulus"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    mean = 6.466  # update 220419 (medium)
    std = 10.367  # update 220419 (medium)


@ex.named_config
def downstream_bulkmodulus_scaled():
    exp_name = "downstream_bulkmodulus_scaled"
    data_root = "/home/data/pretrained_mof/ver4/downstream/20k"
    log_dir = "result_downstream"
    downstream = "bulkmodulus_scaled"
    load_path = "###"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})


@ex.named_config
def downstream_N2diffusivity_dilute():
    exp_name = "downstream_N2diffusivity_dilute"
    data_root = "/home/data/pretrained_mof/coremof/0_diffusivity"
    log_dir = "result_coremof"
    downstream = "N2diffusivity_dilute"
    load_path = "best_ckpt/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 0.000160
    std = 0.000112


@ex.named_config
def downstream_O2diffusivity_dilute():
    exp_name = "downstream_O2diffusivity_dilute"
    data_root = "/home/data/pretrained_mof/coremof/0_diffusivity"
    log_dir = "result_coremof"
    downstream = "O2diffusivity_dilute"
    load_path = "best_ckpt/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 0.000165
    std = 0.000107


@ex.named_config
def downstream_N2uptake():
    exp_name = "downstream_N2uptake"
    data_root = "/home/data/pretrained_mof/coremof/0_diffusivity"
    log_dir = "result_coremof"
    downstream = "N2uptake"
    load_path = "best_ckpt/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 0.358
    std = 0.185


@ex.named_config
def downstream_O2uptake():
    exp_name = "downstream_O2uptake"
    data_root = "/home/data/pretrained_mof/coremof/0_diffusivity"
    log_dir = "result_coremof"
    downstream = "O2uptake"
    load_path = "best_ckpt/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 0.376
    std = 0.203


# solvent removal stability classification
@ex.named_config
def downstream_ssc():
    exp_name = "downstream_ssc"
    data_root = "/home/data/pretrained_mof/coremof/1_stability/ssc"
    log_dir = "result_coremof"
    downstream = "ssc"
    load_path = "best_ckpt/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"classification": 1})
    n_classes = 2


# thermal stability regression
@ex.named_config
def downstream_tsr():
    exp_name = "downstream_tsr"
    data_root = "/home/data/pretrained_mof/coremof/1_stability/tsr"
    log_dir = "result_coremof"
    downstream = "tsr"
    load_path = "best_ckpt/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 361.322
    std = 88.122

@ex.named_config
def downstream_henry_co2():
    exp_name = "downstream_henry_co2"
    data_root = "/home/data/pretrained_mof/coremof/2_henry_co2"
    log_dir = "result_coremof"
    downstream = "henry_co2"
    load_path = "best_ckpt/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = -3.554
    std = 1.120
"""
pretraining (ver 3)
"""


@ex.named_config
def task_ggm():
    exp_name = "task_ggm"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"ggm": 1})

    draw_false_grid = True


@ex.named_config
def task_mtp():
    exp_name = "task_mtp"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1})


@ex.named_config
def task_vfp():
    exp_name = "task_vfp"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"vfp": 1})


@ex.named_config
def task_moc():
    exp_name = "task_moc"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"moc": 1})


@ex.named_config
def task_mtp_ggm():
    exp_name = "task_mtp_ggm"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"ggm": 1, "mtp": 1})

    draw_false_grid = True


@ex.named_config
def task_mtp_bbp():
    exp_name = "task_mtp_bbp"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "bbp": 1})


@ex.named_config
def task_mtp_vfp():
    exp_name = "task_mtp_vfp"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "vfp": 1})


@ex.named_config
def task_mtp_moc():
    exp_name = "task_mtp_moc"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "moc": 1})


@ex.named_config
def task_moc_vfp():
    exp_name = "task_moc_vfp"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"moc": 1, "vfp": 1})


@ex.named_config
def task_mtp_bbp_vfp():
    exp_name = "task_mtp_bbp_vfp"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "bbp": 1, "vfp": 1})


@ex.named_config
def task_mtp_moc_vfp():
    exp_name = "task_mtp_moc_vfp"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "moc": 1, "vfp": 1})


"""
pretraining according to dataset size (ver 3)
"""


@ex.named_config
def task_mtp_moc_vfp_10k():
    exp_name = "task_mtp_moc_vfp_10k"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"
    dataset_size = 10

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "moc": 1, "vfp": 1})


@ex.named_config
def task_mtp_moc_vfp_50k():
    exp_name = "task_mtp_moc_vfp_50k"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"
    dataset_size = 50

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "moc": 1, "vfp": 1})


@ex.named_config
def task_mtp_moc_vfp_100k():
    exp_name = "task_mtp_moc_vfp_100k"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"
    dataset_size = 100

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "moc": 1, "vfp": 1})


@ex.named_config
def task_mtp_moc_vfp_500k():
    exp_name = "task_mtp_moc_vfp_500k"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer"
    dataset_size = 500

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "moc": 1, "vfp": 1})


"""
pretraining according to transformer size (ver 3)
"""


@ex.named_config
def task_mtp_moc_vfp_small():
    exp_name = "task_mtp_moc_vfp_small"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer_small"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    hid_dim = 512
    num_heads = 8
    num_layers = 4
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "moc": 1, "vfp": 1})


@ex.named_config
def task_mtp_moc_vfp_medium():
    exp_name = "task_mtp_moc_vfp_medium"
    data_root = "/home/data/pretrained_mof/ver4/dataset/"
    log_dir = "result_transformer_medium"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    hid_dim = 512
    num_heads = 8
    num_layers = 8
    use_transformer = True
    loss_names = _loss_names({"mtp": 1, "moc": 1, "vfp": 1})
