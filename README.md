# 🌐 Commutative Algebra MOF Training

This repository provides code to train **Gradient Boosting models** on **Category-Algebra (CA) features** of Metal–Organic Frameworks (MOFs).  
Our workflow builds on the benchmark gas property datasets introduced by [Orhan et al. (2021)](https://doi.org/10.1021/acs.jpcc.1c04157).

---

## 📊 Data Sources

We adapted the four datasets from Orhan et al. (2021):  
- [Original repository – MOF-O2N2 GitHub](https://github.com/ibarishorhan/MOF-O2N2/tree/main/mofScripts)  

From this resource you can obtain:  
- **Structures** (CIF files of MOFs)  
- **Property spreadsheets** (Henry’s constants and uptakes for O₂ and N₂)  

We carefully followed the Orhan et al. (2021) protocol to filter, clean, and standardize the data.

---

## 🔄 Structure Conversion (CIF → XYZ)

We provide a helper script to convert raw CIF structures into `.xyz` format for downstream processing:  
👉 [XYZ_Generator.py](https://github.com/CSKhaemba1/MOF-CSCA/blob/main/codes/XYZ_Generator.py)

---

## 🧮 Feature Generation

CA features are generated directly from MOF structures.  
Our construction draws on the algebraic framework of Grayson & Stillman (2002):  

> D. R. Grayson and M. E. Stillman. *Macaulay2: a software system for research in algebraic geometry*, 2002.  

These invariants are adapted to MOFs, creating **category-specific algebraic descriptors** that can be used for machine learning.

---

## 📑 Supported Properties

The training code supports the following property files:

- `HenrysconstantN2.xlsx`  
- `HenrysconstantO2.xlsx`  
- `N2uptakemolkg.xlsx`  
- `O2uptakemolkg.xlsx`  

Each Excel dataset must include:  
- A **MOF ID column** (default: `MOFRefcodes`)  
- A **property value column**  

➡️ These files are available in the [`data`](https://github.com/CSKhaemba1/MOF-CSCA/blob/main/data) folder.  
The MOF IDs must match those in your **features CSV**.

---

## 📂 Features

- **Input file**: Features CSV, generated separately using the CA feature extraction pipeline.  
- **One row = one MOF** with its descriptors.  

---

## ⚙️ Training Protocol

For each property:  
- 🔀 Perform **10 random splits**  
  - 80% training / 10% validation / 10% testing  
- 🏋️ Train **10 Gradient Boosting models** per split  
- 📊 Average predictions across models  
- ✅ Compute metrics (MAE, RMSE, Pearson’s R_p²)  
- 📈 Final score = **mean across 10 splits**
