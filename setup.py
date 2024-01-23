from setuptools import setup

# Dependencies
requirements = [
        "numpy==1.26.2",
        "pandas==2.1.4",
        "plotly==5.18.0",
        "scipy==1.11.4",
        "neurokit2==0.2.7",
        "requests==2.31.0", # neurokit dependency
        "openpyxl==3.1.2", # for excel export
        "xlrd==2.0.1", # for excel import
        ]

setup(
    name="glucopy",
    version="0.1",
    install_requires=requirements,
)