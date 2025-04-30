# Error Troubleshooting

### 1. NumPy Issue  
**Error Message: Numpy Issue**  


**Solution:**  
Run the following commands to reinstall NumPy with a compatible version:
```bash
pip uninstall numpy
pip install numpy==1.26.4
```

---

### 2. Pre-trained Q Model Not Found
**Error Message: assert os.path.exists(self.qeq_site), 'pre-trained Q model does not exist!'**  


**Solution:**  
Check the config.yaml file that you used for building CNMP.

Modify the ['data']['q_model'] path to point to the correct location of pth.tar file in your environment.


**Example build command:**
```bash
sh build.sh --path ./lammps --config ../modelset/config.yaml --checkpoint ../modelset/model.pth.tar
```
Then, open config.yaml and ensure that:
```bash
data:
    q_model: <correct_path_to_q_model.pth.tar>
```
