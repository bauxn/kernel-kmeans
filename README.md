# Kernel K-Means

Implementation of a Kernel K-Means clustering framework in Python. Utilizes Cython to generate efficient C code. <br>
<https://github.com/bauxn/kernel-kmeans> 
## Installation
Installation via Pip does not allow OpenMP. To use OpenMP, see github installation.
#### Pip
Ensure pip is updated, then run: 

> pip install KKMeans

#### Github

Install via:
> git clone https://github.com/bauxn/kernel-kmeans

Then open the project folder and run
> pip install .

Enabling OpenMP: There is a clearly marked line in setup.py that contains the compiler arguments. Before installing, these need 
to be edited so that they contain whichever command your compiler uses to enable OpenMP. There already is a outcommented line which contains the correct arguments (and some additional ones for efficiency) for the msvc compiler.

## Basic Usage

> from KKMeans import KKMeans <br><br>
> kkm = KKMeans(n_clusters=3, kernel="rbf")<br>
> kkm.fit(data)<br>
> print(kkm.labels_)  # shows label for each datapoint <br>
> print(kkm.quality_)  # print quality (default is inertia) of clustering <br><br>
> predictions = kkm.predict(data_to_predict)  # returns labels of points in data_to_predict<br> 


KKMeans also contains the modules _kernels_ (provides functionality to build kernel matrices / calculate kernels), _elkan_ and _lloyd_ which allow to calculate single iterations of the respective algorithms and _quality_, which contains functionality to calculate the silhouette coefficient.
For more elaborate usage consult the thesis on github or the docstrings.

## Limitations
As the computations happen C, in extreme cases overflows and other datatype errors may occur. Critical points are:

| values | datatype |
| --- | --- |
|kernel_matrix| double | 
|n_clusters| long|
|cluster_sizes| long|
|labels|long|

1. The kernel matrix consists of the results of the kernel function, which usually is applied pairwise on the dataset. So ensuring the results are able to fit in a **double** is necessary. 
2. Ensure the number of clusters fits in a long.
3. Ensure the number of points fits in a long.
