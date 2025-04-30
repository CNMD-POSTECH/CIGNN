# <span style="font-size:larger;">CNMP</span>

CNMP is the machine learning potential of the CIGNN (**Charge Integrated Graph Neural Network**) model for amorphous HfO2.

---

### <span style="font-size:larger;">Set Environment & Compile with LAMMPS</span>

1. Activate the environment: `conda activate cignn_env`
2. Clone the stable branch of LAMMPS: `git clone -b stable https://github.com/lammps/lammps.git lammps`
3. Set the environment parameters according to your system in the `build.sh`
4. Modify the config.yaml file to set the appropriate charge model path:
Open the config.yaml file and update the ['data']['q_model'] field to match the correct location of your charge model.
Example (config.yaml):
   ```data:
   q_model: ../modelset/q.pth.tar  # Update this path based on your environment
   ```
5. Build LAMMPS:  
   
   ```bash
   sh build.sh --path <lammps_path> --config <train_config_path> --checkpoint <trained_checkpoint_path>
   ```

   **Example:**

   ```bash
   sh build.sh --path ./lammps --config ../modelset/config.yaml --checkpoint ../modelset/efs.pth.tar
   ```

---

### <span style="font-size:larger;">Run LAMMPS</span>

  ```
  cd cnmp/example
  sh srun.sh
  ```

---

## <span style="font-size:larger;">Error</span>

If you encounter issues during installation or usage, please refer to the `error.txt` file for troubleshooting.  
If the issue persists, feel free to open an issue in this GitHub repository.

---

## <span style="font-size:larger;">References</span>

If you use CNMP in your research, please consider citing the following work:

- Hyo Gyeong Shin, Seong Hun Kim, Eun Ho Kim, Jun Hyeong Gu, Jaeseon Kim, Seon-Gyu Kim, Shin Hyun Kim, Youngjun Park, Hyo Kim, Sunghyun Kim, Duk-Hyun Choe, Donghwa Lee.  
  Charge Integrated Graph Neural Network-based Machine Learning Potential for Amorphous and Non-stoichiometric Hafnium Oxide

---