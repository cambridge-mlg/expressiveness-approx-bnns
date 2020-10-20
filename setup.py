from setuptools import setup, find_packages

requirements = [
    'numpy<1.19.0,>=1.16.4',
    'torch>=1.3.0',
    'scipy',
    'tensorflow>=2.2.0',
    'gpflow>=2.1.0',
    'matplotlib>=3.1.0',
    'sklearn',
    'json_tricks',
]

dependency_links = [ "git+https://github.com/hughsalimbeni/bayesian_benchmarks.git" ]

setup(
    name='inbetween',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
)
