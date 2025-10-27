# Commutative Algebra MOF Training

This repository provides code to train **Gradient Boosting models** on **Category-Algebra (CA) features** of Metal–Organic Frameworks (MOFs).  
The workflow builds on benchmark gas property datasets from [Orhan et al. (2021)](https://doi.org/10.1021/acs.jpcc.1c04157).

---

## Data Sources

We adapted four datasets from Orhan et al. (2021):  
- [MOF-O2N2 GitHub](https://github.com/ibarisorhan/MOF-O2N2)  

These provide:  
- MOF structures (CIF files)  
- Property spreadsheets (Henry’s constants and uptakes for O₂ and N₂)  

The data were filtered, cleaned, and standardized following the published protocol.

---

## Structure Conversion

We include a script to convert CIF files into `.xyz` format:  
[XYZ_Generator.py](https://github.com/CSKhaemba1/MOF-CSCA/blob/main/codes/XYZ_Generator.py)

---

## Feature Generation

The CA features are generated directly from MOF structures.  
They follow the approach of:

> D. R. Grayson and M. E. Stillman. *Macaulay2: a software system for research in algebraic geometry*, 2002.  

These algebraic ideas are adapted to MOFs to create category-specific descriptors for machine learning.

---

## Properties Supported

The training code works with four property files:

- `HenrysconstantN2.xlsx`  
- `HenrysconstantO2.xlsx`  
- `N2uptakemolkg.xlsx`  
- `O2uptakemolkg.xlsx`  

Each file must contain:  
- A MOF ID column (`MOFRefcodes`)  
- A property value column  

The datasets are provided in the [`data`](https://github.com/CSKhaemba1/MOF-CSCA/blob/main/data) folder.

---

## Features

- Input file: a features CSV, generated using the CA feature extraction pipeline.  
- Each row represents one MOF.  

---

## Training

For each property:  
- Perform 10 random splits (80% train / 10% validation / 10% test)  
- Train 10 Gradient Boosting models per split  
- Average predictions across models  
- Compute MAE, RMSE, and Pearson’s R²  
- Report the mean score across splits
