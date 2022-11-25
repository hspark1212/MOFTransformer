import os
import re
from setuptools import setup, find_packages

try:
    import torch
except ImportError:
    raise EnvironmentError('Torch must be installed before install moftransformer')

setup_requires = ['numpy~=1.22.3']

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

with open("README.md", "r") as f:
    long_description=f.read()

extras_require = {
    'docs': ['sphinx', 'livereload', 'myst-parser']
}

with open('moftransformer/__init__.py') as f:
    version = re.search(r"__version__ = '(?P<version>.+)'", f.read()).group('version')


setup(
    name='moftransformer',
    version=version,
    description='moftransformer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yeonghun Kang, Hyunsoo Park',
    author_email='dudgns1675@kaist.ac.kr, phs68660888@gmail.com',
    packages=find_packages(),
    package_data={'moftransformer': ['libs/GRIDAY/*', 'libs/GRIDAY/scripts/*', 'libs/GRIDAY/FF/*',
                                     'assets/*.json', 'examples/dataset/*', 'examples/dataset/**/*',
                                     'examples/raw/*', 'examples/visualize/dataset/*', 'examples/visualize/dataset/test/*']},
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    scripts=[],
    url='https://hspark1212.github.io/MOFTransformer/',
    download_url='https://github.com/hspark1212/MOFTransformer',
    entry_points={'console_scripts':['moftransformer=moftransformer.cli.main:main']},
    python_requires='>=3.8',
)


try:
    import numpy
except (UserWarning, ImportWarning):
    pass
except ImportError:
    os.system('pip install numpy')
