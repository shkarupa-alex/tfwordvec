from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

__VERSION__ = '1.0.0'

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
    install_requires=[
        # https://github.com/tensorflow/tensorflow/issues/7166
        # 'tensorflow>=2.0.0',
        'tfmiss>=0.4.0',
        'nlpvocab>=1.1.5',
    ],
    extras_require={
        'tf_cpu': ['tensorflow>=2.0.0'],
        'tf_gpu': ['tensorflow-gpu>=2.0.0'],
    },
    test_suite='nose.collector',
    tests_require=['nose'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow word2vec char2vec cbow skipgram fasttext',
)
