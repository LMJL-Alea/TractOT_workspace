# Description
TractOT_workspace is [uv](https://docs.astral.sh/uv/) workspace regrouping 3 packages to perform tractography.
The main package is TractOT, which enable tractometry using Optimal Transport (OT) metrics. 

The Phantom package is dedicated to building toy examples. 

The Preprocessing package is to perform preprocessing of DWI and tracts.


# TractOT
TractOT is a package that enable tractometry in a OT fashion way.
The main scripts are MCM_to_tract.py which add mixture of Gaussians to streamlines and coresMAS.py which perform average of Mixture Augmented Streamlines to enable interpatient comparison. 

See the folder examples_sanity_check for small examples of the function and check the Phantom package for a more concrete example.

# Phantom

Phantom is a package that construct toy example where perform tractometry. The examples are built to show that FA based-tractometry is less discriminant than MCM based-tractometry. See the example_pipeline.ipynb note book for an example.


The first trivial example is called Toy+-. And show off an example where FA-tractometry can't make the difference between a crossing and reduction of FA. The second is based on the Disco dataset. It simulates different type of injuries and compare healthy with ill images. The use of the Disco dataset was provided with the courtesy of Emmanuel Caruyer.

Please refer to the Disco information here:
- Original dataset:
https://data.mendeley.com/datasets/fgf86jdfg6/3

- Articles:
https://inserm.hal.science/inserm-03366969/file/1-s2.0-S2352340921007113-main.pdf
https://inria.hal.science/hal-04390592/file/1-s2.0-S1053811923003828-main.pdf


# Preprocessing

Preprocessing is a small package to perform preprocessing of DWI and tracts, for our particular example.


