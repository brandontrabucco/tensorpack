"""Setup script for FasterRCNN."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='FasterRCNN',
    version='0.1',
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('FasterRCNN')],
    description='Tensorpack implementation of Faster RCNN')