import argparse
import sys
import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from LaplaceThickness import LaplaceWallThickness


def read_vtk(filename):
    """Read a VTK polydata file."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def load_and_verify_meshes(input_path):
    """
    Load input file and separate Endo and Epi meshes.
    
    Returns:
        tuple: (endo_poly, epi_poly)
    """
    print(f"Loading input file: {input_path}")
    combined_poly = read_vtk(input_path)
    
    print("Separating Endo and Epi meshes...")
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(combined_poly)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    
    n_regions = conn.GetNumberOfExtractedRegions()
    print(f"Found {n_regions} disconnected components.")
    
    candidates = []
    conn.SetExtractionModeToSpecifiedRegions()
    
    for i in range(n_regions):
        conn.InitializeSpecifiedRegionList()
        conn.AddSpecifiedRegion(i)
        conn.Update()
        
        # Isolate points using CleanPolyData to ensure we have compact point IDs and data
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(conn.GetOutputPort())
        cleaner.Update()
        
        # DeepCopy to ensure independence from pipeline
        poly = vtk.vtkPolyData()
        poly.DeepCopy(cleaner.GetOutput())
        
        # Skip tiny noise
        if poly.GetNumberOfCells() < 100:
            continue

        # Check Mean Region value (Endo > 0, Epi ~ 0)
        arr = poly.GetPointData().GetArray("Regions")
        if arr:
            mean_val = np.mean(vtk_to_numpy(arr))
        else:
            mean_val = 0.0
        
        candidates.append({'poly': poly, 'mean': mean_val, 'cells': poly.GetNumberOfCells()})

    if len(candidates) < 2:
         raise ValueError(f"Expected at least 2 components (Endo, Epi), found {len(candidates)}.")
         
    # Sort by mean: Highest is Endo (Mean > 0), Lowest is Epi (Mean ~ 0)
    candidates.sort(key=lambda x: x['mean'], reverse=True)
    
    endo_poly = candidates[0]['poly']
    epi_poly = candidates[-1]['poly']
    
    endo_mean = candidates[0]['mean']
    epi_mean = candidates[-1]['mean']
    
    print(f"Selected Endo: {endo_poly.GetNumberOfCells()} cells, Mean Regions: {endo_mean:.2f}")
    print(f"Selected Epi:  {epi_poly.GetNumberOfCells()} cells, Mean Regions: {epi_mean:.2f}")
    
    # Verify Endo has Regions array
    if not endo_poly.GetPointData().GetArray("Regions"):
        print("WARNING: 'Regions' array missing from selected Endocardium mesh!")

    return endo_poly, epi_poly


def main():
    parser = argparse.ArgumentParser(description="Calculate cardiac wall thickness using various algorithms.")
    parser.add_argument("input_file", nargs='?', default="endocardium_regions.vtk", help="Path to input VTK file containing both Endo and Epi meshes")
    parser.add_argument("--out", default="results", help="Base name for output files (CSV and VTK will be named as {out}_{algorithm}.{ext})")
    parser.add_argument("--res", type=float, default=1.0, help="Voxel resolution in mm")
    parser.add_argument("--algorithm", default="laplace", help="Algorithm to use (currently: laplace)")
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Load and verify input meshes
    endo_poly, epi_poly = load_and_verify_meshes(args.input_file)
    
    # Generate output filenames based on base name and algorithm
    csv_filename = f"{args.out}_{args.algorithm}.csv"
    vtk_filename = f"{args.out}_{args.algorithm}.vtk"
    
    # Calculate grid bounds
    grid_bounds = LaplaceWallThickness.calculate_grid_bounds(endo_poly, epi_poly)
    
    # Instantiate and execute algorithm
    if args.algorithm.lower() == "laplace":
        calc = LaplaceWallThickness(endo_poly, epi_poly, grid_bounds, args.res)
        success = calc.execute(vtk_filename, csv_filename)
        if not success:
            sys.exit(1)
    else:
        print(f"Error: Unknown algorithm: {args.algorithm}")
        sys.exit(1)


if __name__ == "__main__":
    main()
