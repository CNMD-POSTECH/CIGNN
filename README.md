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
pip install -e .
```

Ensure you update the setup.yaml file to match your system's configuration. Specify the appropriate versions of **torch and torch-scatter** based on your system's environment and hardware. 

If you need to install a specific version of PyTorch, refer to the official PyTorch installation guide and choose the options that are compatible with your system.

---

## <span style="font-size:larger;">Usage</span>

Before running the commands, make sure to set the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=$(pwd)
```

### <span style="font-size:larger;">Train Charge</span>

To train a CIGNN (CPM) model, use the following command:

```bash
cignn-train-q
    --config=./script/run_train_q.yaml
```

### <span style="font-size:larger;">Train Energy & Force</span>

To train a CIGNN (EFPM) model, use the following command:

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

The example datasets are located in the **script/example_dataset** folder, and the example models are located in the **example-1** folder.

### <span style="font-size:larger;">Pretrained Models</span>

Pretrained model files are available for immediate use without additional training:

- Charge model (CPM): `./modelset/q.pth.tar`
- Energy & Force model (EFPM): `./modelset/ef.pth.tar`

---

## <span style="font-size:larger;">Molecular Dynamics</span>

To run MD with a trained model, follow these steps:

1. Navigate to the `cnmp` folder:

2. Read the `README.md` file inside the `cnmp` folder for detailed instructions:

---

## <span style="font-size:larger;">Error</span>

If you encounter issues during installation or usage, please refer to the `error.md` file for troubleshooting.  
If the issue persists, feel free to open an issue in this GitHub repository.

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
