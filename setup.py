import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name='easydl',

    version='0.1',

    author="Stephen Ahmad",

    author_email="privat.ahmad@gmail.com",

    description="An easy to understand deep learning library for educational"
                " purposes.",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/dlahmad/easydl",

    packages=setuptools.find_packages(),

    install_requires=[
        'cupy-cuda100>=6.0.0',
        'numpy>=1.16.4'
    ],

    classifiers=[

     "Programming Language :: Python :: 3",

     "License :: OSI Approved :: MIT License",

     "Operating System :: OS Independent",

    ],
)
