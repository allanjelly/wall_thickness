import argparse
import sys
import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from LaplaceThickness import LaplaceWallThickness
from SimpleThickness import SimpleWallThickness
from RayThickness import RayWallThickness
from FastLaplaceThickness import FastLaplaceWallThickness
from SuperFastLaplaceThickness import SuperFastLaplaceWallThickness


def read_vtk(filename):
    """Read a VTK polydata file."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def load_and_verify_meshes(input_path):
    """
    Load input file and separate Endo and Epi meshes based on 'Regions' scalar.
    
    Endo mesh: polygons with region != 0
    Epi mesh: polygons with region == 0 or not assigned
    
    Returns:
        tuple: (endo_poly, epi_poly)
    """
    print(f"Loading input file: {input_path}")
    combined_poly = read_vtk(input_path)
    
    print("Separating Endo and Epi meshes based on 'Regions' scalar...")
    
    # Get the Regions array from point data
    regions_array = combined_poly.GetPointData().GetArray("Regions")
    
    if regions_array is None:
        raise ValueError("'Regions' array not found in point data. Cannot separate meshes.")
    
    # Convert to numpy for easier manipulation
    regions = vtk_to_numpy(regions_array)
    
    # For each cell, determine if it belongs to endo or epi
    # A cell is endo if ANY of its points has region != 0
    # A cell is epi if ALL of its points have region == 0
    n_cells = combined_poly.GetNumberOfCells()
    endo_mask = np.zeros(n_cells, dtype=bool)
    
    for i in range(n_cells):
        cell = combined_poly.GetCell(i)
        point_ids = cell.GetPointIds()
        
        # Check if any point in this cell has region != 0
        has_nonzero_region = False
        for j in range(point_ids.GetNumberOfIds()):
            point_id = point_ids.GetId(j)
            if regions[point_id] != 0:
                has_nonzero_region = True
                break
        
        endo_mask[i] = has_nonzero_region
    
    epi_mask = ~endo_mask
    
    endo_cell_count = np.sum(endo_mask)
    epi_cell_count = np.sum(epi_mask)
    
    print(f"Found {endo_cell_count} endo cells (region != 0)")
    print(f"Found {epi_cell_count} epi cells (region == 0)")
    
    if endo_cell_count == 0:
        raise ValueError("No endo cells found (region != 0)")
    if epi_cell_count == 0:
        raise ValueError("No epi cells found (region == 0)")
    
    # Extract endo mesh
    endo_poly = extract_cells_by_mask(combined_poly, endo_mask)
    
    # Extract epi mesh
    epi_poly = extract_cells_by_mask(combined_poly, epi_mask)
    
    print(f"Endo mesh: {endo_poly.GetNumberOfPoints()} points, {endo_poly.GetNumberOfCells()} cells")
    print(f"Epi mesh:  {epi_poly.GetNumberOfPoints()} points, {epi_poly.GetNumberOfCells()} cells")
    
    return endo_poly, epi_poly


def extract_cells_by_mask(polydata, mask):
    """
    Extract cells from polydata based on a boolean mask.
    
    Args:
        polydata: Input vtkPolyData
        mask: Boolean numpy array indicating which cells to keep
    
    Returns:
        vtkPolyData with selected cells
    """
    # Create a selection array
    selection_array = numpy_to_vtk(mask.astype(np.uint8))
    selection_array.SetName("Selection")
    
    # Add to cell data temporarily
    polydata.GetCellData().AddArray(selection_array)
    
    # Use threshold filter to extract cells where Selection == 1
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Selection")
    threshold.SetLowerThreshold(1)
    threshold.SetUpperThreshold(1)
    threshold.Update()
    
    # Convert back to polydata
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputConnection(threshold.GetOutputPort())
    geometry_filter.Update()
    
    # Clean up and compact point IDs
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(geometry_filter.GetOutputPort())
    cleaner.Update()
    
    # Create output polydata
    output_poly = vtk.vtkPolyData()
    output_poly.DeepCopy(cleaner.GetOutput())
    
    # Remove the temporary selection array
    output_poly.GetCellData().RemoveArray("Selection")
    polydata.GetCellData().RemoveArray("Selection")
    
    return output_poly


def main():
    parser = argparse.ArgumentParser(description="Calculate cardiac wall thickness using various algorithms.")
    parser.add_argument("input_file", nargs='?', default="endocardium_regions.vtk", help="Path to input VTK file containing both Endo and Epi meshes")
    parser.add_argument("--out", default="results", help="Base name for output files (CSV and VTK will be named as {out}_{algorithm}.{ext})")
    parser.add_argument("--res", type=float, default=0.3, help="Voxel resolution in mm")
    parser.add_argument("--algorithm", default="fastlaplace", help="Algorithm to use (laplace, fastlaplace, superfastlaplace, simple, ray)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers for superfastlaplace (default: CPU count)")
    
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
    
    # Execute selected algorithm
    if args.algorithm.lower() == "laplace":
        grid_bounds = LaplaceWallThickness.calculate_grid_bounds(endo_poly, epi_poly)
        calc = LaplaceWallThickness(endo_poly, epi_poly, grid_bounds, args.res)
        success = calc.execute(vtk_filename, csv_filename)
    elif args.algorithm.lower() == "fastlaplace":
        grid_bounds = FastLaplaceWallThickness.calculate_grid_bounds(endo_poly, epi_poly)
        calc = FastLaplaceWallThickness(endo_poly, epi_poly, grid_bounds, args.res)
        success = calc.execute(vtk_filename, csv_filename)     
    elif args.algorithm.lower() == "superfastlaplace":
        grid_bounds = SuperFastLaplaceWallThickness.calculate_grid_bounds(endo_poly, epi_poly)
        calc = SuperFastLaplaceWallThickness(endo_poly, epi_poly, grid_bounds, args.res)
        success = calc.execute(vtk_filename, csv_filename)             
    elif args.algorithm.lower() == "simple":
        # Note: SimpleThickness does not use grid_bounds or resolution parameters
        calc = SimpleWallThickness(endo_poly, epi_poly)
        success = calc.execute(vtk_filename, csv_filename)
    elif args.algorithm.lower() == "ray":
        # Note: RayThickness does not use grid_bounds or resolution parameters
        calc = RayWallThickness(endo_poly, epi_poly)
        success = calc.execute(vtk_filename, csv_filename)
    else:
        print(f"Error: Unknown algorithm: {args.algorithm}")
        sys.exit(1)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
