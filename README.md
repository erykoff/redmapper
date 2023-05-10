The *red*-sequence *ma*tched-filter *P*robabilistic *Per*colation (redMaPPer)
Cluster Finder
=========

This is the open-source, python version of the *red*-sequence *ma*tched-filter
*P*robabilistic *Per*colation (redMaPPer) cluster finder, originally described
in [Rykoff et al. (2014)](http://adsabs.harvard.edu/abs/2014ApJ...785..104R),
with updates described in [Rozo et
al. (2015)](http://adsabs.harvard.edu/abs/2015MNRAS.453...38R) and [Rykoff et
al. (2016)](http://adsabs.harvard.edu/abs/2016ApJS..224....1R).

Installation
------------

Installing the code is easy, and is supported on Python 2.7 and 3.5+.

```
cd redmapper
python setup.py install
```

Tests
-----
Once installed, it is important to run all the tests:

```
cd redmapper
pytest -v tests/
```

If you encounter any problems, please [file an
issue](https://github.com/erykoff/redmapper/issues).

Dependencies
------------
The following modules are required:
* astropy
* matplotlib
* nose
* pyyaml
* fitsio
* esutil
* numpy
* healpy
* scipy
* future
* [healsparse](https://github.com/lsstdesc/healsparse)

How-To
------
Please see [the redMaPPer How-To](how-to/README.md) for information about
running redMaPPer on your data.
