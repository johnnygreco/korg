#! /usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="korg",
    version='0.1',
    author="Johnny Greco",
    packages=["korg"],
    url="https://github.com/johnnygreco/korg",
    license="MIT"
)
