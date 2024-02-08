from setuptools import setup, find_packages

# Dependencies
requirements = [
        "numpy==1.26.2",
        "pandas==2.1.4",
        "plotly==5.18.0",
        "scipy==1.11.4",
        "astropy==6.0.0",
        "neurokit2==0.2.7",
        "requests==2.31.0", # neurokit dependency
        ]

# Optional dependencies
extras = {
    "excel": ["openpyxl==3.1.2", "xlrd==2.0.1"]
}

setup(
    name = "glucopy",
    version = "0.1.0",
    description = "Python Toolbox for Glycaemic Signal Processing",
    url = "https://github.com/Deigoodle/GlucoPy",
    author = 'Diego Soto Castillo',
    packages = find_packages(),
    install_requires = requirements,
    extras_require = extras,
    python_requires = ">=3.11",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)