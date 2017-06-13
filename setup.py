try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Image Processing',
    'author': 'Daniel Merz',
    'url': 'https://github.com/smba-camera/image_processing',
    'version': '0.1',
    'install_requires': ['numpy', 'nose'],
    'packages': ['image_processing'],
    'scripts': [],
    'name': 'image_processing'
}

setup(**config)
