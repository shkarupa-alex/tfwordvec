from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
from setuptools import setup, find_packages

__VERSION__ = '1.4.1'

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tfwordvec',
    version=__VERSION__,
    description='Word vector estimators built with TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/shkarupa-alex/tfwordvec',
    author='Shkarupa Alex',
    author_email='shkarupa.alex@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.7.0',
    install_requires=Path("requirements.txt").read_text().splitlines(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tfwordvec-vocab=tfwordvec.vocab:main',
            'tfwordvec-train=tfwordvec.train:main',
            'tfwordvec-hub=tfwordvec.hub:main',
            'tfwordvec-export=tfwordvec.export:main',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow word2vec char2vec cbow skipgram fasttext',
)
