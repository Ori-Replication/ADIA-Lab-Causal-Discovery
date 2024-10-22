import os, sys
from setuptools import setup, find_packages

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
        name='Cloud',
        version='1.0',
        description='Causal Discovery method for Reichenbach Proplem in discrete variables',
        author='Masatoshi Kobayashi',
        author_email='kobayashi-masatoshi453@g.ecc.u-tokyo.ac.jp',
        url='https://github.com/Matsushima-lab/Cloud',
        install_requires=read_requirements(),
        keywords=['Causal Discovery'],
        license='MIT',
        packages=find_packages("cloud"),
)


