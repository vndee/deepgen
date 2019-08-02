import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(
    name="deepgen",
    version="0.0.2",
    description="Deep Generative Model",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/vndee/deepgen",
    author="Duy Huynh",
    author_email="hvd.huynhduy@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["deepgen"],
    include_package_data=True,
    install_requires=[],
)
