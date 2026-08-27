# 2024-Cell-Sarate-Hochstetter-Valet

This repository contains the code to run simulations and produce figures for the modelling part (Figures 3 and Supplementary Figure 3) of Sarate, Hochstetter, Valet, et al., Dynamic regulation of tissue fluidity controls skin repair during wound healing, Cell (2024), https://doi.org/10.1016/j.cell.2024.07.031

Details of the model can be found in the Supplemental Theory (Methods S1).

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

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12629079.svg)](https://doi.org/10.5281/zenodo.12629079)

## License

This code is available under a [Attribution-NonCommercial-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/) 

This license strictly prohibits commercial use of the code or derivatives. This includes the prohibition to use the code for the training of AI models for any commercial purpose. Any research, academic, or personal use is permitted.
