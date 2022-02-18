from sacred import Experiment

ex = Experiment("pretrained_mof")


def _loss_names(d):
    ret = {
        "ggm": 0,  # graph grid matching
        "mpp": 0,  # masked patch prediction
        "mtp": 0,  # mof topology prediction
        "vfp": 0,  # (accessible) void fraction prediction
        "moc": 0,  # metal organic classification
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
    use_cgcnn = False  # use CGCNN (crystal graph CNN)
    use_egcnn = False  # use EGCNN (energy grid CNN)
    use_transformer = False  # use graph embedding + vision transformer 3D
    loss_names = _loss_names({})

    # cgcnn
    n_conv = 5  # default of CGCNN=3
    atom_fea_len = 64
    nbr_fea_len = 64  # default : CGCNN = 41

    # egcnn
    egcnn_depth = 18  # 10, 18, 34, 50, 101, 152, 200

    # cgcnn + egcnn
    strategy = 'concat'

    # graph seeting
    # max_atom_len = 1000  # number of maximum atoms in primitive cell
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
    decay_power = 1  # or cosine
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
    batch_size = 256  # desired batch size; for gradient accumulation
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size
    num_gpus = 2
    num_nodes = 1
    load_path = ""
    num_workers = 16  # the number of cpu's core
    precision = 16

    # experiments
    use_only_vit = False
    use_only_mgt = False

    # normalization target
    mean = None
    std = None


@ex.named_config
def env_neuron():
    data_root = "/scratch/x2287a03/dataset"
    per_gpu_batchsize = 16


"""
pretraining with only_vit (ver 3)
"""


@ex.named_config
def vit_task_mtp():
    exp_name = "vit_task_mtp"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
    log_dir = "result_vit"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 16

    # model
    use_only_vit = True
    loss_names = _loss_names({"mtp": 1})


@ex.named_config
def vit_task_vfp():
    exp_name = "vit_task_vfp"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
    log_dir = "result_vit"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 16

    # model
    use_only_vit = True
    loss_names = _loss_names({"vfp": 1})


@ex.named_config
def vit_task_mtp_vfp():
    exp_name = "vit_task_mtp_vfp"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
    log_dir = "result_vit"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 16

    # model
    use_only_vit = True
    loss_names = _loss_names({"mtp": 1, "vfp": 1})


@ex.named_config
def vit_task_mpp():
    exp_name = "vit_task_mpp"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
    log_dir = "result_vit"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 16

    # model
    use_only_vit = True
    loss_names = _loss_names({"mpp": 1})


"""
finetuning with only_vit (ver 3)
"""


@ex.named_config
def downstream_topology():
    exp_name = "downstream_topology_20k"
    data_root = "/home/data/pretrained_mof/ver3/downstream/20k"
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
def downstream_5_scaled():
    exp_name = "downstream_5_scaled_20k"
    data_root = "/home/data/pretrained_mof/ver3/downstream/20k"
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
    data_root = "/home/data/pretrained_mof/ver3/downstream/20k"
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
    data_root = "/home/data/pretrained_mof/ver3/downstream/20k"
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
    data_root = "/home/data/pretrained_mof/qmof/dataset/unrelaxed_data2/20k"
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
def downstream_bulkmodulus():
    exp_name = "downstream_bulkmodulus"
    data_root = "/home/data/pretrained_mof/ver3/downstream/20k"
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

    mean = 4.095
    std = 16.118


"""
pretraining with only_mgt (ver 3)
"""


@ex.named_config
def mgt_task_moc():
    exp_name = "mgt_task_moc"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
    log_dir = "result_mgt"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_only_mgt = True
    loss_names = _loss_names({"moc": 1})


@ex.named_config
def mgt_task_moc_downstream_bandgap():
    exp_name = "downstream_bandgap"
    data_root = "/home/data/pretrained_mof/qmof/dataset/unrelaxed_data2/20k"
    log_dir = "result_mgt_downstream"
    downstream = "bandgap"
    load_path = "/home/hspark8/PycharmProjects/pretrained_mof/best_ckpt/best_mgt_task_moc.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 16

    # model
    use_only_mgt = True
    loss_names = _loss_names({"regression": 1})

    mean = 2.097
    std = 1.088


@ex.named_config
def mgt_task_moc_downstream_bulkmodulus():
    exp_name = "downstream_bulkmodulus"
    data_root = "/home/data/pretrained_mof/ver3/downstream/20k"
    log_dir = "result_mgt_downstream"
    downstream = "bulkmodulus"
    load_path = "/home/hspark8/PycharmProjects/pretrained_mof/best_ckpt/best_mgt_task_moc.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 16

    # model
    use_only_mgt = True
    loss_names = _loss_names({"regression": 1})

    mean = 4.095
    std = 16.118


"""
pretraining (ver 3)
"""


@ex.named_config
def task_ggm():
    exp_name = "task_ggm"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
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
def task_ggm_mtp():
    exp_name = "task_ggm_mtp"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
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
def task_ggm_mtp_moc():
    exp_name = "task_ggm_mtp_moc"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"ggm": 1, "mtp": 1, "moc": 1})

    draw_false_grid = True


@ex.named_config
def task_ggm_mtp_vfp():
    exp_name = "task_ggm_mtp_vfp"
    data_root = "/home/data/pretrained_mof/ver3/dataset/"
    log_dir = "result_transformer"

    # trainer
    max_epochs = 100
    batch_size = 1024
    per_gpu_batchsize = 8

    # model
    use_transformer = True
    loss_names = _loss_names({"ggm": 1, "mtp": 1, "vfp": 1})

    draw_false_grid = True
