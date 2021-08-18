_your zenodo badge here_

# Hadjimichael et al. 2021 JWRPM

**Inferring water scarcity: do large-scale hydrologic and node-based water systems model representations of the Upper Colorado river basin lead to consistent vulnerability insights?**

Antonia Hadjimichael<sup>1\*</sup>, Jim Yoon<sup>2</sup>, Patrick M. Reed<sup>1</sup>, Nathalie Voisin<sup>2</sup>, Wenwei Xu<sup>2</sup>

<sup>1 </sup> School of Civil and Environmental Engineering, Cornell University, Ithaca, NY, USA
<sup>2</sup>Pacific Northwest National Laboratory, Richland, WA, USA

\* corresponding author:  ah986@cornell.edu

## Abstract
Water resources model development and simulation efforts have seen rapid growth in recent decades to aid evaluations and planning around water scarcity and allocation. Models are typically developed by two distinct communities: large-scale hydrologic modelers emphasizing hydro-climatological processes and water systems modelers emphasizing environmental, infrastructural, and institutional features that shape water scarcity at the local basin level. This study assesses whether two representative models from these communities produce consistent insights when evaluating the water scarcity vulnerabilities in the Upper Colorado River Basin within the state of Colorado. Results show that while the regional-scale model (MOSART-WM) can capture the aggregate effect of all water operations in the basin, it underestimates the sub-basin scale variability in specific user’s vulnerabilities. The basin-scale water systems model (StateMod) suggests a larger variance of scarcity across the basin’s water users due to its more detailed accounting of local water allocation infrastructure and institutional processes. This model intercomparison highlights potentially significant limitations of large-scale studies in seeking to evaluate water scarcity and actionable adaptation strategies, as well as ways in which basin-scale water systems model’s information can be used to better inform water allocation and shortage when used in tandem with larger-scale hydrological modeling studies.

## Journal reference
Hadjimichael, A., Yoon, J., Reed, P.M., Voisin, N., Xu, W., Inferring water scarcity: do large-scale hydrologic and node-based water systems model representations of the Upper Colorado river basin lead to consistent vulnerability insights? (submitted to Journal of Water Resources Planning and Management August 2021)

## Code reference



## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| StateMod | 15.0 | https://github.com/OpenCDSS/cdss-app-statemod-fortran | - |
| MOSART-WM | version | https://github.com/IMMM-SFA/wm | https://doi.org/10.5281/zenodo.1225343 |

## Reproduce my experiment

1. Install all package dependencies listed in environment.yml using "conda env create --file environment.yml"
2. Run the following script in the `workflow` directory to re-create this experiment:

| Script Name | Description | How to Run |
| --- | --- | --- |
| `step_one.py` | Script to run the first part of my experiment | `python3 step_one.py -f /path/to/inputdata/file_one.csv` |
| `step_two.py` | Script to run the last part of my experiment | `python3 step_two.py -o /path/to/my/outputdir` |

4. Download and unzip the output data from my experiment [Output data](#output-data)
5. Run the following scripts in the `workflow` directory to compare my outputs to those from the publication

| Script Name | Description | How to Run |
| --- | --- | --- |
| `compare.py` | Script to compare my outputs to the original | `python3 compare.py --orig /path/to/original/data.csv --new /path/to/new/data.csv` |

## Reproduce my figures
Use the scripts found in the `figures` directory to reproduce the figures used in this publication.

| Script Name | Description | How to Run |
| --- | --- | --- |
| `generate_figures.py` | Script to generate my figures | `python3 generate_figures.py -i /path/to/inputs -o /path/to/outuptdir` |
