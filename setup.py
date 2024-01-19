from setuptools import setup

setup(
    name="glucopy",
    version="0.1",
    install_requires=[
        "numpy==1.26.2",
        "pandas==2.1.4",
        "plotly==5.18.0",
        "scipy==1.11.4",
        "neurokit2==0.2.7",
        "requests==2.31.0", # neurokit dependency
    ],
    extras_require={
        "optional": [
            "openpyxl==3.1.2",
            "xlrd==2.0.1",
        ]
    }
)