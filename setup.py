import os
from setuptools import setup, find_packages
import re

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
    print (version)


setup(
    name='moftransformer',
    version='1.0.0',
    description='moftransformer',
    long_description=long_description,
    author='Yeonghun Kang, Hyunsoo Park',
    author_email='dudgns1675@kaist.ac.kr, phs68660888@gmail.com',
    packages=find_packages(),
    package_data={'moftransformer': ['libs/GRIDAY/*', 'libs/GRIDAY/scripts/*', 'libs/GRIDAY/FF/*',
                                     'assets/*.json']},
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    scripts=[],
    download_url='https://github.com/hspark1212/MOFTransformer',
    entry_points={'console_scripts':['moftransformer=moftransformer.cli.main:main']},
    python_requires='>=3.8',
)


try:
    import numpy
except ImportWarning:
    pass
except ImportError:
    os.system('pip install numpy')