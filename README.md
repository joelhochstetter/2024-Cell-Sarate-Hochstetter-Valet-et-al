# 2024-Cell-Sarate-Hochstetter-Valet

This repository contains the code to run simulations and produce figures for Sarate, Hochstetter, Valet, et al, Dynamic regulation of tissue
fluidity controls skin repair during wound healing, Cell 2024.

Details of the model can be found in the Supplemental Theory.

## Relevant folders

- simulator: code to run simulation for Voronoi models with feedback on cell decisions
- fitting: contains codes to fit tissue simulations for experimental data
- experiment: contains raw experimental data that is modelled and shows the processing done
- modelling: contains modelling of different conditions. Contains relevant sub-folders
  - control: Fitting S and S+D model to control data
  - ablation: Fitting S and S+D model to ablation data and computing T1 transition rates





## Dependencies

python3.8.1 used for the paper 

- numpy
- pandas
- matplotlib
- scipy
- joblib
- random
- numba
- sparse
- moviepy
- collections
- sortedcontainers
- tabulate



