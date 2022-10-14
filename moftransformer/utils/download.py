import os
import wget
import tarfile
from pathlib import Path
from moftransformer.database import DEFAULT_PRETRAIN_MODEL_PATH, DEFAULT_COREMOF_PATH, DEFAULT_QMOF_PATH, \
    DEFAULT_FINETUNED_MODEL_PATH, DEFAULT_HMOF_PATH


class DownloadError(Exception):
    pass


def _remove_tmp_file(direc:Path):
    tmp_list = direc.parent.glob('*.tmp')
    for tmp in tmp_list:
        if tmp.exists():
            os.remove(tmp)


def _download_file(link, direc, name='target'):
    if direc.exists():
        print (f'{name} already exists.')
        return
    try:
        print(f'\n====Download {name} =============================================\n')
        filename = wget.download(link, out=str(direc))
    except KeyboardInterrupt:
        _remove_tmp_file(direc)
        raise
    except Exception as e:
        _remove_tmp_file(direc)
        raise DownloadError(e)
    else:
        print (f'\n====Successfully download : {filename}=======================================\n')


def download_pretrain_model(direc=None, ):
    if not direc:
        direc = Path(DEFAULT_PRETRAIN_MODEL_PATH)
    else:
        direc = Path(direc)

    if not direc.suffix:
        if not direc.exists():
            direc.mkdir(parents=True, exist_ok=True)
        direc = direc / 'pretrained_model.ckpt'
    elif direc.suffix == '.ckpt':
        if not direc.parent.exists():
            direc.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f'direc must be path for directory or ~.ckpt, not {direc}')

    link = 'https://figshare.com/ndownloader/files/37511767'
    name = 'pretrain_model'
    _download_file(link, direc, name)


def download_finetuned_model(direc=None, ):
    if not direc:
        direc = Path(DEFAULT_FINETUNED_MODEL_PATH)
        if not direc.exists():
            direc.mkdir(parents=True, exist_ok=True)
    else:
        direc = Path(direc)
        if not direc.suffix:
            if not direc.exists():
                direc.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f'direc must be path for directory, not {direc}')

    link = 'https://figshare.com/ndownloader/files/37621520'
    name = 'finetuned_bandgap'
    _download_file(link, direc / 'finetuned_bandgap.ckpt', name)

    link = 'https://figshare.com/ndownloader/files/37622693'
    name = 'finetuned_h2_uptake'
    _download_file(link, direc / 'finetuned_h2_uptake.ckpt', name)


def download_coremof(direc=None, remove_tarfile=False):
    if not direc:
        direc = Path(DEFAULT_COREMOF_PATH)
    else:
        direc = Path(direc)

    if direc.suffix:
       raise ValueError(f'direc must be directory, not {direc}')
    elif (direc/'raw/GIRGUL_clean.grid').exists():
        print ('CoREMOF database is already existed')
        return
    elif not direc.exists():
        direc.mkdir(parents=True, exist_ok=True)
    direc = direc/'coremof.tar.gz'

    link = 'https://figshare.com/ndownloader/files/37511746'
    name = 'coremof_database'
    _download_file(link, direc, name)

    print(f'\n====Unzip : {direc}===============================================\n')
    with tarfile.open(direc) as f:
        f.extractall(path=direc.parent)

    print(f'\n====Unzip successfully: {direc}===============================================\n')

    if remove_tarfile:
        os.remove(direc)


def download_qmof(direc=None, remove_tarfile=False):
    if not direc:
        direc = Path(DEFAULT_QMOF_PATH)
    else:
        direc = Path(direc)

    if direc.suffix:
       raise ValueError(f'direc must be directory, not {direc}')
    elif (direc/'raw/JORCOK_FSR.grid').exists():
        print ('QMOF database is already existed')
        return
    elif not direc.exists():
        direc.mkdir(parents=True, exist_ok=True)
    direc = direc/'qmof.tar.gz'


    link = 'https://figshare.com/ndownloader/files/37511758'
    name = 'qmof_database'
    _download_file(link, direc, name)

    print(f'\n====Unzip : {direc}===============================================\n')
    with tarfile.open(direc) as f:
        f.extractall(path=direc.parent)

    print(f'\n====Unzip successfully: {direc}===============================================\n')

    if remove_tarfile:
        os.remove(direc)



def download_hmof(direc=None, remove_tarfile=False):
    if not direc:
        direc = Path(DEFAULT_HMOF_PATH)
    else:
        direc = Path(direc)

    if direc.suffix:
       raise ValueError(f'direc must be directory, not {direc}')
    elif (direc/'downstream_release/train_raspa_100bar.json').exists():
        print ('hMOF database is already existed')
        return
    elif not direc.exists():
        direc.mkdir(parents=True, exist_ok=True)
    direc = direc/'hmof.tar.gz'

    link = 'https://figshare.com/ndownloader/files/37511755'
    name = 'hmof_database'
    _download_file(link, direc, name)

    print(f'\n====Unzip : {direc}===============================================\n')
    with tarfile.open(direc) as f:
        f.extractall(path=direc.parent)

    print(f'\n====Unzip successfully: {direc}===============================================\n')

    if remove_tarfile:
        os.remove(direc)
