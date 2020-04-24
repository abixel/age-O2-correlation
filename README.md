# age-O2-correlation

This script explores the detectability of a correlation between the ages of habitable planets and whether or not they have oxygen ("age-O2 correlation", Bixel & Apai, under review).

Feel free to email A. Bixel (abixel@email.arizona.edu) with questions and comments!

This script was written and tested in Python 3.6; other Python versions may not be compatible.

### Dependencies:

`matplotlib`
`numpy`
`scipy`
`scikit-image`
`tqdm`

### Description of modules:

`generate.py`: contains functions to generate the results for a selected parameterization, sample, and statistical test (see below)
`models.py`: includes 4 possible parameterizations of the age-O2 correlation and 5 choices for the target sample
`tests.py`: defines statistical tests used to detect the age-O2 correlation
`plots.py`: generates Figures 1-3 from the referenced paper - the simulation results must be generated first!

### Quick start guide:

From the command line, run `generate.py` to generate results for each of the 11 scenarios explored in the paper. The "result" for each scenario is a grid of p-values corresponding to each tested sample size and value of f_life (the fraction of observed planets which are inhabited).

Next, run `plots.py` to generate the plots for Figures 1 - 3.





