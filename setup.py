import setuptools
from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements

with open('README.md', 'r') as fh:
    long_description = fh.read()

requirements = parse_requirements(
    filename='requirements.txt',
    session=PipSession()
)

setuptools.setup(
    name='python-polar-coding',

    version='0.0.1',

    description='Polar coding implementation in Python',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/fr0mhell/python-polar-coding',

    author='Grigory Timofeev',

    author_email='t1m0feev.grigorij@gmail.com',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='polar codes, fec, simulation',

    packages=setuptools.find_packages(),

    python_requires='>=3.6, <4',

    install_requires=[req.requirement for req in requirements],

    project_urls={  # Optional
        'Bug Report': 'https://github.com/fr0mhell/python-polar-coding/issues',
        'Source': 'https://github.com/fr0mhell/python-polar-coding',
    },
)
