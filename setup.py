#!/usr/bin/env python
from pathlib import Path
from setuptools import setup, find_packages

long_description = Path("README.md").read_text()

setup(
    name='nomad',
    url='https://github.com/Watts-Lab/nomad',
    version='0.2.0',
    author='Francisco Barreras, Thomas Li, Federico Delussu, Andres Mondragon',
    author_email='fbarrer@sas.upenn.edu, thomli@sas.upenn.edu, fedde@dtu.dk, amt00@sas.upenn.edu',
    description='NOMAD provides a repository of processing and analysis methods for GPS mobility data, centralizing tools necessary for large-scale analyses. This repository will serve the dual function of facilitating research for first-time users and enhancing the replicability and robustness of existing methodologies. By collecting many such methods in a single place and documenting their assumptions and robustness metrics, NOMAD aims to enhance methodological transparency and replicability of research.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=find_packages(include=['nomad', 'nomad.*']),
    python_requires='>=3.9',

    install_requires=[
        'pandas',
        'geopandas',
        'numpy',
        'datetime',
        'shapely',
        'matplotlib',
        'networkx',
        'osmnx',
        'pygeohash',
        'libgeohash',
        'funkybob',
        'scipy',
        'pyarrow',
        's3fs',
        'h3',
        'pydeck'
    ],

    extras_require={
        'spark': [
            'pyspark>=3.4.4,<4',
            'sedona>=1.4.1'
        ]
    },

    package_data={'nomad': ['data/*', 'data/**/*']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
