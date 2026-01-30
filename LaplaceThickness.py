import vtk
import numpy as np
import time
import sys

# Try to import scipy, handle availability
try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    print("Error: This script requires 'scipy'. Please install it via pip: pip install scipy")
    sys.exit(1)

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


class LaplaceWallThickness:
    @staticmethod
    def calculate_grid_bounds(endo_poly, epi_poly):
        """Calculate combined bounds for both meshes with padding."""
        b1 = endo_poly.GetBounds()
        b2 = epi_poly.GetBounds()
        
        return [
            min(b1[0], b2[0]) - 1.0, max(b1[1], b2[1]) + 1.0,
            min(b1[2], b2[2]) - 1.0, max(b1[3], b2[3]) + 1.0,
            min(b1[4], b2[4]) - 1.0, max(b1[5], b2[5]) + 1.0
        ]

    def __init__(self, endo_poly, epi_poly, grid_bounds, resolution=1.0):
        """
        Initialize Laplace wall thickness calculator.
        
        Args:
            endo_poly: vtkPolyData for endocardium mesh
            epi_poly: vtkPolyData for epicardium mesh
            grid_bounds: List of 6 floats [xmin, xmax, ymin, ymax, zmin, zmax]
            resolution: Voxel resolution in mm
        """
        self.endo_poly = endo_poly
        self.epi_poly = epi_poly
        self.grid_bounds = grid_bounds
        self.resolution = resolution
        self.dims = None
        self.spacing = None
        self.origin = None
        
        # Grid states
        self.STATE_BLOOD = 0  # Endo (Potential 0)
        self.STATE_WALL = 1   # Myocardium (Solve for potential)
        self.STATE_OUTSIDE = 2 # Epi and beyond (Potential 1)
        
        # Solution
        self.potential_field = None
        self.thickness_map = None

    def voxelize_domain(self):
        """
        Create a structured grid and classify voxels into Blood, Wall, Outside.
        """
        print(f"Voxelizing domain with resolution {self.resolution}mm...")
        
        # 1. Setup Grid Dimensions
        self.spacing = [self.resolution] * 3
        self.origin = [self.grid_bounds[0], self.grid_bounds[2], self.grid_bounds[4]]
        
        dims = [
            int(np.ceil((self.grid_bounds[1] - self.grid_bounds[0]) / self.resolution)),
            int(np.ceil((self.grid_bounds[3] - self.grid_bounds[2]) / self.resolution)),
            int(np.ceil((self.grid_bounds[5] - self.grid_bounds[4]) / self.resolution))
        ]
        self.dims = dims
        print(f"Grid dimensions: {dims} ({dims[0]*dims[1]*dims[2]} voxels)")

        # 2. Convert PolyData to Stencil (Binary Mask)
        # Endo Mask (1 inside endo, 0 outside)
        endo_mask = self._polydata_to_numpy_mask(self.endo_poly, dims, self.origin, self.spacing)
        
        # Epi Mask (1 inside epi, 0 outside)
        epi_mask = self._polydata_to_numpy_mask(self.epi_poly, dims, self.origin, self.spacing)
        
        # 3. Combine to create domain labels
        # Initialize as OUTSIDE (2)
        self.domain_labels = np.full(dims, self.STATE_OUTSIDE, dtype=np.uint8)
        
        # Wall is Inside Epi AND Outside Endo
        # Logic:
        # If inside Endo -> BLOOD (0)
        # Else if inside Epi -> WALL (1)
        # Else -> OUTSIDE (2)
        
        # Set Wall
        self.domain_labels[epi_mask > 0] = self.STATE_WALL
        
        # Set Blood (overwrites wall if endo is inside epi, which it should be)
        self.domain_labels[endo_mask > 0] = self.STATE_BLOOD
        
        wall_count = np.sum(self.domain_labels == self.STATE_WALL)
        print(f"Voxel counts: Blood={np.sum(self.domain_labels == self.STATE_BLOOD)}, "
              f"Wall={wall_count}, Outside={np.sum(self.domain_labels == self.STATE_OUTSIDE)}")
        return wall_count

    def _polydata_to_numpy_mask(self, polydata, dims, origin, spacing):
        """Convert vtkPolyData to a numpy boolean/binary mask via vtkPolyDataToImageStencil"""
        # Create empty image data to define the bounds/spacing
        white_image = vtk.vtkImageData()
        white_image.SetSpacing(spacing)
        white_image.SetDimensions(dims)
        white_image.SetOrigin(origin)
        # Allocate is needed for Extent calculation but we don't need scalars
        white_image.SetExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
        
        # PolyData to Image Stencil
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
        pol2stenc.Update()
        
        # Convert stencil to image
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(white_image)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff() # Check what the default is. Usually stencil is 1 inside.
        # Actually vtkImageStencil takes an image and masks it. 
        # Better: export mask directly from stencil data.
        
        # Easier approach: Use vtkPolyDataToImageStencil -> vtkImageStencilToImage
        # But vtkImageStencilToImage is not always available in older VTK.
        
        # Let's use the standard "Stencil to Image" flow
        # Create an image of all zeros, set to 1 where stencil is active
        image = vtk.vtkImageData()
        image.SetDimensions(dims)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
        # Fill with 0
        from vtk.util import numpy_support
        arr = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
        arr.fill(0)
        
        # Apply stencil logic manualy or using specific filter
        # It's surprisingly annoying to just get the binary mask from stencil in python vtk without vtkImageStencilToImage
        
        # Alternative: vtkSelectEnclosedPoints (slower but robust) is a point cloud filter.
        # Given the grid size, this might be slow for millions of points.
        
        # Let's try vtkImageStencilToImage if available (VTK 8.2+)
        try:
            stenc2img = vtk.vtkImageStencilToImage()
            stenc2img.SetInputConnection(pol2stenc.GetOutputPort())
            stenc2img.SetInsideValue(1)
            stenc2img.SetOutsideValue(0)
            stenc2img.SetOutputScalarType(vtk.VTK_UNSIGNED_CHAR)
            stenc2img.Update()
            res = stenc2img.GetOutput()
        except AttributeError:
            # Fallback for older VTK: create a blank image, use vtkImageStencil to "paint"
            # Paint background 0
            # Paint stencil 1
            # Requires distinct background/foreground setup
            background = vtk.vtkImageData()
            background.SetDimensions(dims)
            background.SetSpacing(spacing)
            background.SetOrigin(origin)
            background.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
            scalar_arr = numpy_support.vtk_to_numpy(background.GetPointData().GetScalars())
            scalar_arr.fill(0) # Outside
            
            stencil_filter = vtk.vtkImageStencil()
            stencil_filter.SetInputData(background)
            stencil_filter.SetStencilConnection(pol2stenc.GetOutputPort())
            stencil_filter.SetBackgroundValue(1) # Inside (Assuming ReverseStencil is Off means stencil covers inside)
            # Wait, SetBackgroundValue sets the value for the area NOT covered by stencil if ReverseStencil is Off
            # By default ReverseStencil is Off. The Stencil covers the "Region". 
            # vtkImageStencil passes the Input through where Stencil is active, and replaces with BackgroundValue where NOT active.
            # So if we want Inside=1, Outside=0:
            # Input image should be all 1s. Background Value should be 0.
            scalar_arr.fill(1) # Input all 1s
            stencil_filter.SetBackgroundValue(0) # Outside stencil becomes 0
            stencil_filter.Update()
            res = stencil_filter.GetOutput()
            
        # Convert to numpy
        sc = res.GetPointData().GetScalars()
        arr = vtk_to_numpy(sc)
        return arr.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0) # VTK is ZYX, we want XYZ or verify

    def solve_laplace(self):
        """
        Solve Ax=b for gradient field.
        A: Laplacian operator
        b: Boundary conditions
        x: Potential
        """
        print("Constructing Laplacian matrix...")
        # Grid dimensions
        nx, ny, nz = self.dims
        n_voxels = nx * ny * nz
        
        # Identify linear indices of Wall voxels
        # numpy uses C-order (last index changes fastest), but we need to match how we flattened/accessed the grid
        # self.domain_labels is [dx, dy, dz] i.e. [x, y, z] if we were careful
        # But VTK export was ZYX. Let's stick to using the transposed array which is [x, y, z]
        # X is 0..dims[0], Y is 0..dims[1], Z is 0..dims[2]
        
        # Flatten labels for linear indexing
        # order='F' means first index changes fastest (Column-major like Matlab/Fortran). 
        # order='C' means last index changes fastest (Row-major like C).
        # We need to be consistent. Let's use 'C' (default) -> index = x*ny*nz + y*nz + z ??? No.
        # In 'C' (default numpy): index = x * (dimY * dimZ) + y * dimZ + z
        
        labels_flat = self.domain_labels.ravel() 
        
        # Indices of unknowns
        unknown_indices = np.where(labels_flat == self.STATE_WALL)[0]
        n_unknowns = len(unknown_indices)
        
        if n_unknowns == 0:
            print("No wall voxels found! Check mesh overlap and normal orientation.")
            return

        # Map global voxel index -> local unknown index
        # -1 indicates not an unknown (boundary or outside)
        global_to_local = np.full(n_voxels, -1, dtype=np.int32)
        global_to_local[unknown_indices] = np.arange(n_unknowns)
        
        # Build Sparse Matrix (7-point stencil)
        # We invoke loops or vectorized construction. Vectorized is better.
        
        # Strides for neighbors in flattened array
        # array is [x, y, z], shape (nx, ny, nz)
        # stride_z = 1
        # stride_y = nz
        # stride_x = ny * nz
        stride_z = 1
        stride_y = nz
        stride_x = ny * nz
        
        rows = []
        cols = []
        data = []
        rhs = np.zeros(n_unknowns, dtype=np.float64)
        
        # Helper to add connections
        # Neighbor offsets in flat array
        offsets = [-stride_x, stride_x, -stride_y, stride_y, -stride_z, stride_z]
        
        print(f"Building linear system for {n_unknowns} unknowns...")
        
        # For each unknown voxel i
        #   A[i, i] = 6
        #   For each neighbor j:
        #     if j is Wall: A[i, local(j)] = -1
        #     if j is Blood (0): Add (1 * 0) to RHS -> 0 contribution
        #     if j is Outside/Epi (1): Add (1 * 1) to RHS -> +1 to RHS
        
        # Vectorized Approach:
        # Create an array of unknown_indices
        # Calculate neighbor indices
        u_idx = unknown_indices
        
        # Diagonal elements (always 6 in 3D 7-point stencil, unless boundary of grid)
        # We treat grid boundaries as Neumann (derivative=0) or just ignore since we have padding.
        # With padding, all wall voxels are safely inside.
        
        center_diag = np.ones(n_unknowns) * 6.0
        
        for offset in offsets:
            neighbors = u_idx + offset
            
            # Use valid neighbors (strictly inside grid limits check usually needed, 
            # but with padding we can skip if we padded enough)
            
            neighbor_labels = labels_flat[neighbors]
            neighbor_locals = global_to_local[neighbors]
            
            # Connection to other Wall voxels
            is_wall = (neighbor_labels == self.STATE_WALL)
            # Add -1 to A[row, col]
            # row: 0..n_unknowns (range(n_unknowns))
            # col: global_to_local[neighbor]
            
            valid_wall_mask = is_wall
            valid_rows = np.arange(n_unknowns)[valid_wall_mask]
            valid_cols = neighbor_locals[valid_wall_mask]
            
            rows.append(valid_rows)
            cols.append(valid_cols)
            data.append(np.full(len(valid_rows), -1.0))
            
            # Boundary Conditions
            # If neighbor is OUTSIDE (Epi), we subtract (-1 * V_epi) from LHS => Add V_epi to RHS
            # V_epi = 1.0
            is_epi = (neighbor_labels == self.STATE_OUTSIDE)
            rhs[is_epi] += 1.0
            
            # If neighbor is BLOOD (Endo), V_endo = 0.0
            # rhs += 0.0 (noop)

        # Combine sparse parts
        rows.append(np.arange(n_unknowns))
        cols.append(np.arange(n_unknowns))
        data.append(center_diag)
        
        row_arr = np.concatenate(rows)
        col_arr = np.concatenate(cols)
        data_arr = np.concatenate(data)
        
        A = sp.csr_matrix((data_arr, (row_arr, col_arr)), shape=(n_unknowns, n_unknowns))
        
        print("Solving linear system...")
        start_t = time.time()
        x = spsolve(A, rhs)
        print(f"Solved in {time.time() - start_t:.2f}s")
        
        # Map back to full grid
        self.potential_field = np.zeros(n_voxels, dtype=np.float32)
        # Set boundaries
        self.potential_field[labels_flat == self.STATE_OUTSIDE] = 1.0
        self.potential_field[labels_flat == self.STATE_BLOOD] = 0.0
        # Set calculated wall
        self.potential_field[unknown_indices] = x
        
        # Reshape to 3D
        self.potential_field = self.potential_field.reshape(self.dims)

    def calculate_thickness(self):
        """
        Compute gradients and integrate streamlines from Endo to Epi.
        """
        print("Calculating thickness map...")
        
        # 1. Compute Gradients
        # Gradient of potential field
        # grad is (3, nx, ny, nz)
        grad = np.gradient(self.potential_field, self.spacing[0], self.spacing[1], self.spacing[2])
        grad_x, grad_y, grad_z = grad[0], grad[1], grad[2]
        
        # Normalize to get direction
        norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        norm[norm == 0] = 1.0 # Avoid div/0
        
        dir_x = grad_x / norm
        dir_y = grad_y / norm
        dir_z = grad_z / norm
        
        # Interpolators for direction field
        # Prepare grid coordinates
        x = np.linspace(self.origin[0], self.origin[0] + (self.dims[0]-1)*self.spacing[0], self.dims[0])
        y = np.linspace(self.origin[1], self.origin[1] + (self.dims[1]-1)*self.spacing[1], self.dims[1])
        z = np.linspace(self.origin[2], self.origin[2] + (self.dims[2]-1)*self.spacing[2], self.dims[2])
        
        interp_dx = RegularGridInterpolator((x, y, z), dir_x, bounds_error=False, fill_value=0)
        interp_dy = RegularGridInterpolator((x, y, z), dir_y, bounds_error=False, fill_value=0)
        interp_dz = RegularGridInterpolator((x, y, z), dir_z, bounds_error=False, fill_value=0)
        
        interp_pot = RegularGridInterpolator((x, y, z), self.potential_field, bounds_error=False, fill_value=1)

        # 2. Iterate over Endo Mesh Points
        endo_points = vtk_to_numpy(self.endo_poly.GetPoints().GetData())
        n_points = len(endo_points)
        thicknesses = np.zeros(n_points)
        
        print(f"Tracing streamlines for {n_points} points...")
        
        # Integration parameters
        step_size = self.resolution * 0.5 
        max_steps = int(20.0 / step_size) # Max 20mm thickness
        
        for i, start_pt in enumerate(endo_points):
            # Verify we start near 0 potential
            # pot = interp_pot([start_pt])[0]
            
            curr_pos = start_pt.copy()
            length = 0.0
            
            for step in range(max_steps):
                # Sample direction
                d = np.array([
                    interp_dx([curr_pos])[0],
                    interp_dy([curr_pos])[0],
                    interp_dz([curr_pos])[0]
                ])
                
                # Check for zero gradient (could happen inside blood or far outside)
                d_norm = np.linalg.norm(d)
                if d_norm < 0.001:
                    break
                    
                # Take step
                step_vec = d * step_size
                curr_pos += step_vec
                length += step_size
                
                # Check potential
                pot = interp_pot([curr_pos])[0]
                if pot >= 0.95: # Reached Epi
                    break
            
            thicknesses[i] = length
            
            if i % 1000 == 0:
                print(f"  Processed {i}/{n_points} points...", end='\r')
        
        print(f"  Processed {n_points}/{n_points} points. Done.")
        
        # 3. Add thickness to polydata
        arr = numpy_to_vtk(thicknesses)
        arr.SetName("LaplaceThickness")
        
        pd = self.endo_poly.GetPointData()
        pd.AddArray(arr)
        pd.SetActiveScalars("LaplaceThickness")
        
        self.endo_poly.GetPointData().SetActiveScalars("LaplaceThickness")
        

    def analyze_regions(self, csv_filename="endocardium_wall_thickness.csv"):
        print("\nAnalyzing regions...")
        
        pd = self.endo_poly.GetPointData()
        regions_arr = pd.GetArray("Regions")
        if not regions_arr:
            print("Error: 'Regions' array not found on endocardium mesh.")
            return

        thickness_arr = pd.GetArray("LaplaceThickness")

        if not thickness_arr:
             print("Error: 'LaplaceThickness' array not found.")
             return

        regions = vtk_to_numpy(regions_arr)
        thicknesses = vtk_to_numpy(thickness_arr)
        
        # Define region names (copied from LAsegmenter.py)
        extended_region_names = [
            'Background', 'RSPV', 'LSPV', 'RIPV', 'LIPV', 'MA', 'LAA',
            'Posterior_Wall', 'Roof', 'Inferior_Wall', 'Lateral_Wall',
            'Septal_Wall', 'Anterior_Wall', 'RSPV_Ostium', 'LSPV_Ostium',
            'RIPV_Ostium', 'LIPV_Ostium'
        ]
        
        # Define categories matches
        def get_category(rid):
            if rid in [7, 8, 9, 10, 11, 12]: return "Wall"
            if rid in [13, 14, 15, 16]: return "Ostium"
            if rid in [1, 2, 3, 4, 5, 6]: return "PV/Special"
            return "Unknown"

        import csv
        results_data = []

        print(f"\n{'Region':<20} {'Avg(mm)':<10} {'Std(mm)':<10} {'Points':<10}")
        print("-" * 60)
        
        # Iterate regions 1 to 16
        for rid in range(1, 17):
            mask = (regions == rid)
            count = np.sum(mask)
            
            if count > 0:
                vals = thicknesses[mask]
                
                # Filter outliers (simple filtering for now, comparable to logic in thickness_analysis.py)
                # Keep values > 0.1 and < 10.0 ??
                # The prompt asks to calculate based on Laplace.
                # Usually Laplace is robust, but let's just report full stats first or apply mild filtering.
                # reference CSV has 'Discard_TooClose', etc.
                # Here we just have the raw Laplace result.
                # Let's filter implausible values (e.g. 0 or very large)
                
                valid_mask = (vals >= 0.1) & (vals <= 15.0) # slightly loose bounds
                valid_vals = vals[valid_mask]
                
                discard_count = len(vals) - len(valid_vals) # Roughly "total discards"
                
                if len(valid_vals) > 0:
                    avg_t = np.mean(valid_vals)
                    std_t = np.std(valid_vals)
                else:
                    avg_t = 0
                    std_t = 0

                name = extended_region_names[rid] if rid < len(extended_region_names) else f"Region_{rid}"
                category = get_category(rid)
                
                print(f"{name:<20} {avg_t:<10.4f} {std_t:<10.4f} {len(valid_vals):<10}")
                
                results_data.append({
                    'Region_ID': rid,
                    'Region_Name': name,
                    'Category': category,
                    'Avg_Thickness_mm': round(avg_t, 4),
                    'Std_Dev_mm': round(std_t, 4),
                    'Valid_Vertices': len(valid_vals),
                    'Discard_TooClose': 0, # Placeholder
                    'Discard_TooFar': 0, # Placeholder
                    'Discard_BadNormal': 0, # Placeholder
                    'Discard_Outlier': discard_count,
                    'Total_Vertices': count
                })
        
        # Save to CSV
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['Region_ID', 'Region_Name', 'Category', 'Avg_Thickness_mm', 'Std_Dev_mm',
                             'Valid_Vertices', 'Discard_TooClose', 'Discard_TooFar',
                             'Discard_BadNormal', 'Discard_Outlier', 'Total_Vertices']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_data)
            print(f"\n✓ Results saved to {csv_filename}")
        except Exception as e:
            print(f"✗ Failed to save CSV: {e}")

    def save_output(self, output_path):

        print(f"Saving result to {output_path}")
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(self.endo_poly)
        writer.Write()

    def execute(self, vtk_filename, csv_filename):
        """
        Execute the Laplace thickness calculation pipeline.
        
        Args:
            vtk_filename: Path to output VTK file
            csv_filename: Path to output CSV file
        
        Returns:
            bool: True if execution successful, False otherwise
        """
        # Execute algorithm
        count = self.voxelize_domain()
        
        if count > 0:
            self.solve_laplace()
            self.calculate_thickness()
            self.save_output(vtk_filename)
            self.analyze_regions(csv_filename)
            return True
        else:
            print("Error: Voxelization failed or empty wall domain.")
            return False
