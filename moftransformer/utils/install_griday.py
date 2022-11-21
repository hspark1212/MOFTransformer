import subprocess
from pathlib import Path
from moftransformer.utils.prepare_data import GRIDAY_PATH

__all__ = ['install_griday', 'uninstall_griday']


class InstallationError(Exception):
    def __init__(self, error_message=None):
        self.error_message = error_message

    def __str__(self):
        if self.error_message:
            return f"Installation Error : {self.error_message}"
        return "Installation Error"

def _install_make():
    print ('=== Download gcc=9.5.0 =========================================================')
    ps = subprocess.run('conda install -c conda-forge gcc=9.5.0 -y'.split())
    if ps.returncode:
        raise InstallationError(ps.stderr)
    else:
        print ('=== Successfully download =======================================================')
    print ('=== Download gxx=9.5.0 =========================================================')
    ps = subprocess.run('conda install -c conda-forge gxx=9.5.0 -y'.split())
    if ps.returncode:
        raise InstallationError(ps.stderr)
    else:
        print ('=== Successfully download =======================================================')
    print ('=== Download make=4.2.1 ===============================================================')
    ps = subprocess.run('conda install -c anaconda make=4.2.1 -y'.split())
    if ps.returncode:
        raise InstallationError(ps.stderr)
    else:
        print ('=== Successfully download =======================================================')


def _make_griday():
    dir_griday = Path(GRIDAY_PATH).parent.parent
    if not dir_griday.exists():
        raise InstallationError(f'Invalid path specified : {dir_griday}')
    print('=== Install GRIDAY ==============================================================')
    ps = subprocess.run(['make'], cwd=dir_griday)
    if ps.returncode:
        raise InstallationError(ps.stderr)
    ps = subprocess.run(['make'], cwd=dir_griday/'scripts')
    if ps.returncode:
        raise InstallationError(ps.stderr)
    print('=== Successfully download =======================================================')
    if Path(GRIDAY_PATH).exists():
        print(f'GRIDAY is installed to {dir_griday}')
    else:
        raise InstallationError(f'GRIDAY is not installed. Please try again.')


def install_griday():
    """
    Installation a GRIDAY which calculates the energy grid for prepare-data.py
    Original code : https://github.com/Sangwon91/GRIDAY.git

    :return: None
    """
    _install_make()
    _make_griday()


def uninstall_griday():
    """
    Remove a GRIDAY which calculates the energy grid for prepare-data.py
    Original code : https://github.com/Sangwon91/GRIDAY.git
    :return:
    """

    dir_griday = Path(GRIDAY_PATH).parent.parent
    if not dir_griday.exists():
        raise InstallationError(f'Invalid path specified : {dir_griday}')
    print('=== Uninstall GRIDAY ============================================================')
    ps = subprocess.run(['make', 'clean'], cwd=dir_griday)
    if ps.returncode:
        raise InstallationError(ps.stderr)
    ps = subprocess.run(['make', 'clean'], cwd=dir_griday / 'scripts')
    if ps.returncode:
        raise InstallationError(ps.stderr)
    print('=== Successfully remove  =========================================================')

    if not Path(GRIDAY_PATH).exists():
        print(f'GRIDAY is uninstalled')
    else:
        raise InstallationError()


if __name__ == '__main__':
    install_griday()