# CA-only MOF Training

This repository provides code to train Gradient Boosting models on **Category-Algebra (CA) features** of MOFs, using the benchmark gas property datasets originally introduced by Orhan et al. (2021).
---
## Data Sources

The four datasets used here were **adapted from Orhan et al. (2021)**  
[Descriptor-Based Prediction of Gas Adsorption in MOFs](https://doi.org/10.1021/acs.jpcc.1c04157).

- Original repository: [MOF-O2N2 GitHub](https://github.com/ibarishorhan/MOF-O2N2/tree/main/mofScripts)  
- From this repo you can obtain:
  - **Structures** (CIF files of MOFs)
  - **Property spreadsheets** (Henry’s constants, uptakes, self-diffusion coefficients)

We followed the CSTL (2025) protocol to filter, clean, and standardize these data.

---

## Supported Properties

This code supports the four focus properties:

- `HenrysconstantN2.xlsx`  
- `HenrysconstantO2.xlsx`  
- `N2uptakemolkg.xlsx`  
- `O2uptakemolkg.xlsx`  

Each Excel file must contain:
- A MOF ID column (default: `MOFRefcodes`)  
- One column with the property values.

The MOF IDs should match those in your **features CSV**.
---

## Features

- **Input features CSV**: generated separately using our CA feature extraction pipeline.  
- Each row = one MOF.  
---

## Training Protocol

For each property:
- Perform **10 random splits**  
  - 80% train, 10% validation, 10% test  
- For each split, train **10 models**   
- Average predictions across 10 models (per split)  
- Compute metrics  
- Final result = **mean across 10 splits** 

---

