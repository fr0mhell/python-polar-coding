import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='python-polar-coding',
    version='1.0.0',
    author='Grigory Timofeev',
    author_email='t1m0feev.grigorij@gmail.com',
    description='Polar coding implementation in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fr0mhell/python-polar-coding',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numba==0.44.0',
        'numpy==1.16.4',
        'pycrc==1.21',
        'anytree==2.7.1',
        'fastcache==1.1.0',
        'pymongo==3.9.0',
    ],
)
