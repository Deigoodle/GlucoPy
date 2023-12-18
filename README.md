# Python-Glycemic-Signal-Processing-Toolbox

## Public Datasets

- [Awesome-CGM](https://github.com/irinagain/Awesome-CGM)

  This is a collection of links to publicly available continuous glucose monitoring (CGM) data.

- [OhioT1DM](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html)

  The OhioT1DM dataset is available to researchers interested in improving the health and wellbeing of people with type 1 diabetes. It contains 8 weeks worth of data for each     of 12 people with type 1 diabetes. a Data Use Agreement (DUA) is required.

- [ShanghaiT1DM and ShanghaiT2DM](https://www.nature.com/articles/s41597-023-01940-7#ref-CR40)
  
  ShanghaiT1DM and ShanghaiT2DM are publicly available Datasets for research purposes, they contain Type 1 (n = 12) and Type 2 (n = 100) diabetic patients in Shanghai, China.

- [CG Map](https://github.com/ayya-keshet/CGMap)

  A reference resource for characterization of CGM data collected from more than 7,000 non-diabetic individuals, aged 40-70 years.

## Collection of Python Signal Processing Library Repositories

- [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics/tree/master)
- [StatsModels](https://github.com/statsmodels/statsmodels/)
- [BioSPPy - Biosignal Processing in Python](https://github.com/PIA-Group/BioSPPy)
- [Type-2-Diabetes-Prediction-Using-Short-PPG-Signals-and-Physiological-Characteristics](https://github.com/chirathyh/clardia---Type-2-Diabetes-Prediction-Using-Short-PPG-Signals-and-Physiological-Characteristics-)
- [splearn: Python Signal Processing](https://github.com/jinglescode/python-signal-processing)
- [spm1d: One-Dimensional Statistical Parametric Mapping in Python and MATLAB](https://github.com/0todd0000/spm1d/)
- [NeuroKit2: The Python Toolbox for Neurophysiological Signal Processing](https://github.com/neuropsychology/NeuroKit)
- [PyGSP: Graph Signal Processing in Python](https://github.com/epfl-lts2/pygsp)

## File Structure Proposal
```
├── .gitignore
├── README.md
└── src/
    ├── read_file/
        ├── read_csv.py
        ├── read_xlsx.py
        └── read_txt.py
    ├── format_data/
        ├── format.py
        ├── format.py
        └── format.py
    ├── metrics/
        ├── metrics1.py
        ├── metrics2.py
        └── metrics3.py
    └── plot/
        ├── ex_plot1.py
        ├── ex_plot2.py
        └── ex_plot3.py
```
