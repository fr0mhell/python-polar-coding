import setuptools
from pip._internal.download import PipSession
from pip._internal.req import parse_requirements

with open('README.md', 'r') as fh:
    long_description = fh.read()

requirements = parse_requirements(
    filename='requirements/production.txt',
    session=PipSession()
)

setuptools.setup(
    name='python-polar-coding',
    version='0.2.1',
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
    install_requires=[str(requirement.req) for requirement in requirements],
)
