# Atrial wall thickness
This is a couple of Python scripts that are used for calculating the heart's atrium thickness using different algorithms.
Keep in mind that the required resolution of the output files impacts the script's performance very heavily.
At 0.5mm resolution example atrium voxelization creates 4.8mln voxels.
At 0.25mm number of voxels rises to 38mln
At 0.1mm it becomes 590mln
Memory and CPU requirements escalate very quickly

The input for the scripts is *.vtk file consisting both endocardium and epicardium meshes.
Epicardium vertexes should also be assigned to atrium segments (pvs, walls etc.) to allow the calculation of segmented results.

Usage: 
    python main.py input_file [--out Outfile] [--res Resolution] [--algorithm] 

main.py is a facade behind which plugins implementing different algorithm are hidden.
Currently implemented algorithms:
  - simple (SimpleThickness.py) - based on the principle of closest vertex
  - ray (RayThickness.py) - calculates the thickness of the mesh by raytracing normals from the internal mesh
  - laplace (LaplaceThickness.py) - Voxelizes the mesh with given resolution, solves the Laplace equation, calculates streamlines, and wall thickness.
    Bear in mind it uses Numpy(https://numpy.org/) and Scipy (https://scipy.org/), and is not very fast for meshes over 400k vertices at resolutions lower than 0.3mm
  - fastlaplace (FastLaplaceThickness.py) - as above, but also using pyamg (https://pyamg.readthedocs.io/en/latest/index.html) and numba jit compiler (https://numba.pydata.org/) to speed up calculation.
    This one is pretty fast - calculating thickness for 2 meshes of 200k vertices with a resolution of 0.3mm takes around 11 seconds on a decent laptop.
  - superfastlaplace (SuperFastLaplaceThickness.py) - builds upon the previous by accelerating calculations by using a GPU. Requires nvidia based graphic card, Cuda toolchain, and Cupy (https://cupy.dev/) libraries.
    Bear in mind Cupy, Cuda and Graphic Driver have to be in compatible version.
    This one is capable of solving the Laplace equations at a resolution of up to 0.1mm on 16mb PC in some 40-60sec.

Output:
  - Script outputs data in both:
    - .csv format (Calculated thickness per atrium region) and
    - .vtk format (Calculated thickness on a per/point basis) - ready for visualization
Add.info:
  - batch.py - does what's on a tin. Finds .vtk files and processes them as a batch.


    
