from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

__VERSION__ = '1.3.1'

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
    python_requires='>=3.6.0',
    install_requires=[
        'tensorflow>=2.5.0',
        'tensorflow-addons>=0.13.0',
        'tensorflow-hub>=0.12.0',
        'tfmiss>=0.13.6',
        'nlpvocab>=1.2.0',
        'gensim>=3.8.1'
    ],
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
