#! /usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="korg",
    version='v0.1',
    author="Johnny Greco",
    author_email="jgreco@astro.princeton.edu",
    packages=["korg"],
    url="https://github.com/johnnygreco/korg",
    license="MIT"
)
