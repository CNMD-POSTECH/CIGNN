# <span style="font-size:larger;">CIGNN</span>

CIGNN (**Charge Integrated Graph Neural Network**) is a machine learning potential model designed to predict atomic charges, energies, and forces. 

This model enables the generation of customized potentials tailored for specific materials, offering high flexibility and accuracy for advanced materials simulations.

![CNMD Banner Image](https://github.com/CNMD-POSTECH/CIGNN/blob/main/.github/CNMD.png?raw=true)

---

## <span style="font-size:larger;">Installation and Requirements</span>

### <span style="font-size:larger;">Install from PyPI</span>

This is the recommended method for installing CIGNN:

```bash
git clone https://github.com/CNMD-POSTECH/CIGNN
cd CIGNN
```

```bash
conda env create -f setup.yaml
conda activate cignn_env
pip install --upgrade pip
pip install .
```

Ensure you update the setup.yaml file to match your system's configuration. Specify the appropriate versions of **torch and torch-scatter** based on your system's environment and hardware. 

If you need to install a specific version of PyTorch, refer to the official PyTorch installation guide and choose the options that are compatible with your system.

---

## <span style="font-size:larger;">Usage</span>

### <span style="font-size:larger;">Train</span>

To train a CIGNN model, use the following command:

```bash
cignn-train
    --config=./script/run_train.yaml
```

### <span style="font-size:larger;">Prediction</span>

To run prediction with a trained model, use the following command:

```bash
cignn-predict
    --config=./script/run_predict.yaml
```

Make sure to adjust the YAML file to fit your specific dataset and requirements.

The example datasets are located in the **script/example_dataset** folder, and the trained atomic charge prediction models are located in the **script/example_model** folder.

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

This repository includes contributions from Hyo Gyeong Shin, Seong Hun Kim, Donghwa Lee, and other collaborators.

<a href="https://github.com/CNMD-POSTECH/GB-CGCNN/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CNMD-POSTECH/GB-CGCNN" />
</a>