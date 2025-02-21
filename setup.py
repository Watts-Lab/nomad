#!/usr/bin/env python

from setuptools import setup

setup(
    name='nomad',
    url='https://github.com/Watts-Lab/nomad',
    version='0.0.1',
    author='Thomas Li, Francisco Barreras, Andrés Mondragón',
    author_email='thomli@sas.upenn.edu, fbarrer@sas.upenn.edu, amt00@sas.upenn.edu',
    description='Placeholder',
    packages=['nomad'],
    install_requires=[
        'pandas',
        'geopandas',
        'numpy',
        'datetime',
        'pyspark',
        'shapely',
        'matplotlib',
        'funkybob'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)

