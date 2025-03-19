#!/usr/bin/env python

from setuptools import setup

setup(
    name='nomad',
    url='https://github.com/Watts-Lab/nomad',
    version='0.0.1',
    author='Francisco Barreras, Thomas Li, Federico Delussu, Andres Mondragon',
    author_email='fbarrer@sas.upenn.edu, thomli@sas.upenn.edu, fedde@dtu.dk',
    description='NOMAD provides a repository of processing and analysis methods for GPS mobility data, centralizing tools necessary for large-scale analyses. This repository will serve the dual function of facilitating research for first-time users and enhancing the replicability and robustness of existing methodologies. By collecting many such methods in a single place and documenting their assumptions and robustness metrics, NOMAD aims to enhance methodological transparency and replicability of research.',
    packages=['nomad'],
    install_requires=[
        'pandas',
        'numpy',
        'datetime',
        'pyspark',
        'shapely',
        'matplotlib',
        'networkx',
        'pygeohash',
        'libgeohash',
        'funkybob',
        'scipy',
        'pyarrow',
        's3fs'
    ],
    package_data={'nomad': ['data/*', 'data/**/*']
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

