This is the github repository for my bachelor thesis.<br>
<br>
To compile cython code:<br>
1.: <br>
    Replace /openmp in first (commented) line of _lloyd_iter.pyx with the argument your compiler needs to enable openmp.<br>
    gcc: -fopenmp<br>
    visual studio: /openmp<br>
<br>
2.:<br>
    run:<br>
    python setup.py build_ext --setup<br>
<br>
Then just run main file as usual
