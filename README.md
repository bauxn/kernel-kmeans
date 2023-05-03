This is the github repository for my bachelor thesis.

To compile cython code:
1.: 
    Replace /openmp in first (commented) line of _lloyd_iter.pyx with the argument your compiler needs to enable openmp.
    gcc: -fopenmp
    visual studio: /openmp

2.:
    run:
    python setup.py build_ext --setup

Then just run main file as usual