from setuptools import setup, find_packages

setup(
    name="your_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    author="Francesco Turci",
    author_email="f.turci@bristol.ac.uk",
    description="Supporting code for the Bristol Complex Disordered Matter course.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fturci/comp-complex-disordered-matter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)