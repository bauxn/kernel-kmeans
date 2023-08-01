This is the github repository for my bachelor thesis.<br>
<br>
simple installation via "pip install KKMeans" <br>
<br>
To enable openMP, clone the project and open setup.py, you will find a list compile_args. The dcython argument prevents cython from transpling the assertions to c (-> do not remove).
Add whatever arguments your compiler needs to compile with openMP (and all other args you like) <br>
there are default arguments for msvc and gcc listed. <br>
<br>
when finished editing setup.py (or being content without openMP), install with "pip install ." in the root directory (the same where setup.py resides) <br>
testes for windows11 and ubuntu 22.04.2 <br>
upgrade pip before installation, as pyproject.toml without setup.cfg is a rather new standard <br>
