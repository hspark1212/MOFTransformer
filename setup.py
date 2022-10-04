from setuptools import setup, find_packages

setup_requires = ['numpy~=1.22.3']

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

with open("README.md", "r") as f:
    long_description=f.read()

setup(
    name='moftransformer',
    version='1.0.0',
    description='moftransformer',
    long_description=long_description,
    author='Yeonghun Kang, Hyunsoo Park',
    author_email='dudgns1675@kaist.ac.kr, phs68660888@gmail.com',
    packages=find_packages(),
    package_data={'moftransformer': ['libs/GRIDAY/*', 'libs/GRIDAY/scripts/*', 'libs/GRIDAY/FF/*']},
    install_requires=install_requires,
    setup_requires=setup_requires,
    scripts=[],
    download_url='https://github.com/hspark1212/MOFTransformer',
    python_requires='>=3.8',
)
