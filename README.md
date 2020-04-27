# age-O2-correlation

This code explores the detectability of a correlation between the ages of habitable planets and whether or not they have oxygen ("age-O2 correlation", Bixel & Apai, under review).

Feel free to email A. Bixel (abixel@email.arizona.edu) with questions and comments!

The code was written and tested in Python 3.6; other Python versions may not be compatible.

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


### Quick start guide

From the command line, run `generate.py` to generate the p-value grids used to create the figures in the paper. These will be saved as .pkl files in the `results` output folder.

Next, run `plots.py` to generate the three figures from the paper. These will be saved as .pdf files under `output`.

By default, the plots will be fairly coarse. For better results, set `N_runs` to ~1000 and `N_grid` in `generate.py`. This will take much longer to run.





