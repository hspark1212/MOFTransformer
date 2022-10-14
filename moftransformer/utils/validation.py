from moftransformer.config import _loss_names
from pytorch_lightning.utilities import _IS_INTERACTIVE
import warnings


class ConfigurationError(Exception):
    pass


def _set_loss_names(loss_name):
    if isinstance(loss_name, list):
        d = {k: 1 for k in loss_name}
    elif isinstance(loss_name, str):
        d = {loss_name: 1}
    elif isinstance(loss_name, dict):
        d = loss_name
    elif loss_name is None:
        d = {}
    else:
        raise ConfigurationError(f'loss_name must be list, str, or dict, not {type(loss_name)}')
    return _loss_names(d)


def _set_valid_batchsize(_config):
    if isinstance(num_gpus := _config['num_gpus'], list):
        num_gpus = len(num_gpus)

    per_gpu_batchsize = _config['batch_size'] // num_gpus

    _config['per_gpu_batchsize'] = per_gpu_batchsize
    warnings.warn("'Per_gpu_batchsize' is larger than 'batch_size'.\n"
          f" Adjusted to per_gpu_batchsize to {per_gpu_batchsize}")


def _check_valid_num_gpus(_config):
    if isinstance(num_gpus := _config['num_gpus'], list):
        num_gpus = len(num_gpus)

    if num_gpus > _config['batch_size']:
        raise ConfigurationError('Number of gpus must be smaller than batch_size. '
                         f'num_gpus : {num_gpus}, batch_size : {_config["batch_size"]}')

    if _IS_INTERACTIVE and num_gpus > 1:
        pass
        raise ConfigurationError('The interactive environment (ex. jupyter notebook) does not supports multi-GPU. '
                                 'If you want to use multi-gpu, make *.py file and run.')


def get_valid_config(_config):
    # set loss_name to dictionary
    _config['loss_names'] = _set_loss_names(_config['loss_names'])

    # check_valid_num_gpus
    _check_valid_num_gpus(_config)

    # Batch size must be larger than gpu_per_batch
    if _config['batch_size'] < _config['per_gpu_batchsize']:
        _set_valid_batchsize(_config)

    return _config