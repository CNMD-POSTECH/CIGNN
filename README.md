# <span style="font-size:larger;">CIGNN</span>

CIGNN (**Charge Integrated Graph Neural Network**) is a machine learning potential model designed to predict atomic charges, energies, and forces. 

This model enables the generation of customized potentials tailored for specific materials, offering high flexibility and accuracy for advanced materials simulations.

![CNMD Banner Image](https://github.com/CNMD-POSTECH/CIGNN/blob/main/.github/CNMD.png?raw=true)

---

## <span style="font-size:larger;">Installation and Requirements</span>

Ensure your system meets the following requirements before installation:

- **Python**: >= 3.7
- **[PyTorch](https://pytorch.org/)**: >= 1.12  
To install PyTorch, follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) and select options suitable for your system configuration.

### <span style="font-size:larger;">Install from PyPI</span>

This is the recommended method for installing CIGNN:

```bash
pip install --upgrade pip
pip install cignn-torch
```

### <span style="font-size:larger;">Install from Source</span>

If you prefer to use the latest source code:

```bash
git clone https://github.com/CNMD-POSTECH/CIGNN.git
pip install ./CIGNN
```

---

## <span style="font-size:larger;">Usage</span>

### <span style="font-size:larger;">Train</span>

To train a CIGNN model, use the following command:

```bash
python ./scripts/run_train.py \
    --config=./scripts/run_train.yaml
```

### <span style="font-size:larger;">Prediction</span>

To run prediction with a trained model, use the following command:

```bash
python ./scripts/run_predict.py \
    --config=./scripts/run_predict.yaml
```

Make sure to adjust the YAML file to fit your specific dataset and requirements.

---

## <span style="font-size:larger;">References</span>

If you use CIGNN in your research, please consider citing the following work:


- Hyo Gyeong Shin, Seong Hun Kim, Eun Ho Kim, Jun Hyeong Gu, Jaeseon Kim, Seon-Gyu Kim, Shin Hyun Kim, Youngjun Park, Hyo Kim, Sunghyun Kim, Duk-Hyun Choe, Donghwa Lee.  
  Charge Integrated Graph Neural Network-based Machine Learning Potential for Amorphous and Non-stoichiometric Hafnium Oxide

---

## <span style="font-size:larger;">Contact</span>

For inquiries, feel free to reach out to:
- Hyo Gyeong Shin ([hyogyeong@postech.ac.kr](mailto:hyogyeong@postech.ac.kr))

For bug reports or feature requests, please open an issue on [GitHub Issues](https://github.com/CNMD-POSTECH/CIGNN/issues).

---

## <span style="font-size:larger;">License</span>

CIGNN is distributed under the [MIT License](MIT.md).

---

## <span style="font-size:larger;">Contributors</span>

This repository includes contributions from Hyo Gyeong Shin, Seong Hun Kim, and other collaborators.

<a href="https://github.com/CNMD-POSTECH/GB-CGCNN/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CNMD-POSTECH/GB-CGCNN" />
</a>




