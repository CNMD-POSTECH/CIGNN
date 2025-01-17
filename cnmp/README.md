# <span style="font-size:larger;">CNMP</span>

CNMP is the machine learning potential of the CIGNN (**Charge Integrated Graph Neural Network**) model.

---

### <span style="font-size:larger;">Set Environment & Compile with LAMMPS</span>

1. Activate the environment: `conda activate cignn_env`
2. Clone the stable branch of LAMMPS: `git clone -b stable https://github.com/lammps/lammps.git lammps`
3. Set the environment parameters according to your system in the `build.sh`.
4. Build LAMMPS: `sh build.sh --path=../lammps --config=<config_path> --checkpoint=<checkpoint_path>`

---

### <span style="font-size:larger;">Run LAMMPS</span>

  ```
  cd cnmp/example
  sh run.sh
  ```

---

## <span style="font-size:larger;">References</span>

If you use CNMP in your research, please consider citing the following work:

- Hyo Gyeong Shin, Seong Hun Kim, Eun Ho Kim, Jun Hyeong Gu, Jaeseon Kim, Seon-Gyu Kim, Shin Hyun Kim, Youngjun Park, Hyo Kim, Sunghyun Kim, Duk-Hyun Choe, Donghwa Lee.  
  Charge Integrated Graph Neural Network-based Machine Learning Potential for Amorphous and Non-stoichiometric Hafnium Oxide

---