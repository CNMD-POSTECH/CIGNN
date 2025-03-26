# **Charge Integrated GNN-based MLP (CNMP) Dataset**

This repository contains datasets tailored for the development of machine learning potentials for **CNMP**.

---

## **Dataset Structure**

The datasets are organized as follows:

- **EF Dataset**: Contains structure, atom_number, lattice, energy and force data.
- **Q Dataset**: Contains structure, atom_number, lattice, energy, force and charge data.

---

## **Dataset Split**

The dataset is split into training, validation, and test sets using an **8:1:1 ratio**, ensuring a balanced distribution of samples.

| Type | Dataset   | EFS (Energy & Force)              | Charge (Subset of EF) |
|------|-----------|-----------------------------------|-----------------------|
| NVT  | **EF**    | 345,874 samples (1 fs extraction) | 0 samples             |
| NVT  | **Q**     | 20,980 samples (10 fs extraction) | 20,980 samples        |

---

## **EF Dataset Details**

### 🔹 Atom Count Distribution
| Number of Atoms | Sample Count |
|------------------|---------------|
| 96               | 60,000        |
| 95               | 44,488        |
| 94               | 100,000       |
| 93               | 141,386       |

### 🔹 Temperature Distribution
| Temperature (K) | Sample Count |
|------------------|---------------|
| 300              | 35,000        |
| 500              | 35,001        |
| 700              | 30,000        |
| 900              | 30,000        |
| 1100             | 40,000        |
| 1300             | 30,000        |
| 1700             | 5,000         |
| 1900             | 40,000        |
| 2300             | 35,873        |
| 2500             | 5,000         |
| 2700             | 30,000        |
| 2900             | 30,000        |

---

## **Q Dataset (Charge Data) Details**

- Extracted every **10 fs**
- Temperature distribution is not explicitly recorded, but assumed to cover a wide thermal range

### 🔹 Atom Count Distribution
| Number of Atoms | Sample Count |
|------------------|---------------|
| 96               | 3,500         |
| 95               | 2,980         |
| 94               | 5,500         |
| 93               | 9,000         |

### 🔹 Temperature Distribution
| Temperature (K) | Sample Count |
|------------------|---------------|
| 300              | 3,000         |
| 500              | 6,500         |
| 1000             | 1,980         |
| 2000             | 2,500         |
| 3000             | 3,500         |
| 5000             | 3,500         |