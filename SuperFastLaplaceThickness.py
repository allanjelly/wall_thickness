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

# Try to import pyamg for faster solving
try:
    import pyamg
    PYAMG_AVAILABLE = True
except ImportError:
    PYAMG_AVAILABLE = False
    print("Warning: pyamg not available. Install for faster solving: pip install pyamg")

# Try to import numba for JIT compilation
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Install for faster streamlines: pip install numba")

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpsla
    CUPY_AVAILABLE = True
    # Test if GPU is actually available
    try:
        cp.cuda.Device(0).compute_capability
    except:
        CUPY_AVAILABLE = False
        print("Warning: CuPy installed but no CUDA GPU detected")
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: cupy not available. Install for GPU acceleration: pip install cupy-cuda12x")

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def trilinear_interpolate(array, x, y, z, origin, spacing, dims):
        """
        Fast trilinear interpolation using Numba.
        
        Args:
            array: 3D numpy array to interpolate from
            x, y, z: Point coordinates in world space
            origin: Grid origin (ox, oy, oz)
            spacing: Grid spacing (sx, sy, sz)
            dims: Grid dimensions (nx, ny, nz)
        
        Returns:
            Interpolated value
        """
        # Convert world coordinates to grid indices
        gx = (x - origin[0]) / spacing[0]
        gy = (y - origin[1]) / spacing[1]
        gz = (z - origin[2]) / spacing[2]
        
        # Get integer indices
        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        k0 = int(np.floor(gz))
        
        # Check bounds
        if i0 < 0 or i0 >= dims[0] - 1 or j0 < 0 or j0 >= dims[1] - 1 or k0 < 0 or k0 >= dims[2] - 1:
            return 0.0
        
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # Get fractional parts
        fx = gx - i0
        fy = gy - j0
        fz = gz - k0
        
        # Trilinear interpolation
        c000 = array[i0, j0, k0]
        c001 = array[i0, j0, k1]
        c010 = array[i0, j1, k0]
        c011 = array[i0, j1, k1]
        c100 = array[i1, j0, k0]
        c101 = array[i1, j0, k1]
        c110 = array[i1, j1, k0]
        c111 = array[i1, j1, k1]
        
        # Interpolate along x
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        
        # Interpolate along y
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        
        # Interpolate along z
        return c0 * (1 - fz) + c1 * fz

    @numba.jit(nopython=True, cache=True)
    def trace_streamline_numba(start_pt, dir_x, dir_y, dir_z, potential, 
                               origin, spacing, dims, step_size, max_steps):
        """
        Trace a streamline from start point to epicardium using Numba.
        
        Returns:
            Thickness in mm
        """
        curr_x = start_pt[0]
        curr_y = start_pt[1]
        curr_z = start_pt[2]
        length = 0.0
        
        for step in range(max_steps):
            # Sample direction at current position
            dx = trilinear_interpolate(dir_x, curr_x, curr_y, curr_z, origin, spacing, dims)
            dy = trilinear_interpolate(dir_y, curr_x, curr_y, curr_z, origin, spacing, dims)
            dz = trilinear_interpolate(dir_z, curr_x, curr_y, curr_z, origin, spacing, dims)
            
            # Check for zero gradient
            d_norm = np.sqrt(dx*dx + dy*dy + dz*dz)
            if d_norm < 0.001:
                break
            
            # Take step
            curr_x += dx * step_size
            curr_y += dy * step_size
            curr_z += dz * step_size
            length += step_size
            
            # Check potential
            pot = trilinear_interpolate(potential, curr_x, curr_y, curr_z, origin, spacing, dims)
            if pot >= 0.95:
                break
        
        return length


class SuperFastLaplaceWallThickness:
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
        
        # Timing stats
        self.timing_stats = {}

    def voxelize_domain(self):
        """
        Create a structured grid and classify voxels into Blood, Wall, Outside.
        """
        t_start_total = time.time()
        print(f"\n{'='*70}")
        print(f"VOXELIZING DOMAIN (resolution={self.resolution}mm)")
        print(f"{'='*70}")
        
        # 1. Setup Grid Dimensions
        t_start = time.time()
        self.spacing = [self.resolution] * 3
        self.origin = [self.grid_bounds[0], self.grid_bounds[2], self.grid_bounds[4]]
        
        dims = [
            int(np.ceil((self.grid_bounds[1] - self.grid_bounds[0]) / self.resolution)),
            int(np.ceil((self.grid_bounds[3] - self.grid_bounds[2]) / self.resolution)),
            int(np.ceil((self.grid_bounds[5] - self.grid_bounds[4]) / self.resolution))
        ]
        self.dims = dims
        print(f"Grid dimensions: {dims} ({dims[0]*dims[1]*dims[2]} voxels)")
        self.timing_stats['grid_setup'] = time.time() - t_start
        print(f"  ✓ Grid setup: {self.timing_stats['grid_setup']:.3f}s")

        # 2. Convert PolyData to Stencil (Binary Mask)
        # Endo Mask (1 inside endo, 0 outside)
        t_start = time.time()
        endo_mask = self._polydata_to_numpy_mask(self.endo_poly, dims, self.origin, self.spacing)
        self.timing_stats['endo_voxelization'] = time.time() - t_start
        print(f"  ✓ Endo voxelization: {self.timing_stats['endo_voxelization']:.3f}s")
        
        # Epi Mask (1 inside epi, 0 outside)
        epi_mask = self._polydata_to_numpy_mask(self.epi_poly, dims, self.origin, self.spacing)
        
        # 3. Combine to create domain labels
        # Initialize as OUTSIDE (2)
        t_start = time.time()
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
        self.timing_stats['domain_labeling'] = time.time() - t_start
        print(f"  ✓ Domain labeling: {self.timing_stats['domain_labeling']:.3f}s")
        print(f"\nVoxel counts: Blood={np.sum(self.domain_labels == self.STATE_BLOOD)}, "
              f"Wall={wall_count}, Outside={np.sum(self.domain_labels == self.STATE_OUTSIDE)}")
        
        self.timing_stats['voxelize_total'] = time.time() - t_start_total
        print(f"\n{'='*70}")
        print(f"VOXELIZATION COMPLETE: {self.timing_stats['voxelize_total']:.3f}s")
        print(f"{'='*70}")
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
        t_start_total = time.time()
        print(f"\n{'='*70}")
        print(f"SOLVING LAPLACE EQUATION")
        print(f"{'='*70}")
        
        # Grid dimensions
        t_start = time.time()
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
        
        self.timing_stats['setup_unknowns'] = time.time() - t_start
        print(f"  ✓ Setup unknowns ({n_unknowns:,} unknowns): {self.timing_stats['setup_unknowns']:.3f}s")
        
        # Build Sparse Matrix (7-point stencil) - INCREMENTAL CONSTRUCTION
        t_start = time.time()
        
        # Strides for neighbors in flattened array
        # array is [x, y, z], shape (nx, ny, nz)
        stride_z = 1
        stride_y = nz
        stride_x = ny * nz
        
        rhs = np.zeros(n_unknowns, dtype=np.float32)
        
        # Neighbor offsets in flat array
        offsets = [-stride_x, stride_x, -stride_y, stride_y, -stride_z, stride_z]
        
        # Incremental COO matrix construction - build per offset to avoid large list memory
        u_idx = unknown_indices
        
        # Start with diagonal
        print(f"  Building diagonal...")
        center_diag = np.ones(n_unknowns, dtype=np.float32) * 6.0
        row_indices = np.arange(n_unknowns, dtype=np.int32)
        col_indices = np.arange(n_unknowns, dtype=np.int32)
        
        A = sp.coo_matrix((center_diag, (row_indices, col_indices)), 
                          shape=(n_unknowns, n_unknowns), dtype=np.float32)
        
        # Process each offset incrementally
        for i, offset in enumerate(offsets):
            print(f"  Building offset {i+1}/6...")
            neighbors = u_idx + offset
            
            neighbor_labels = labels_flat[neighbors]
            neighbor_locals = global_to_local[neighbors]
            
            # Connection to other Wall voxels
            is_wall = (neighbor_labels == self.STATE_WALL)
            valid_wall_mask = is_wall
            valid_rows = np.arange(n_unknowns, dtype=np.int32)[valid_wall_mask]
            valid_cols = neighbor_locals[valid_wall_mask]
            
            # Build COO for this offset
            offset_data = np.full(len(valid_rows), -1.0, dtype=np.float32)
            offset_coo = sp.coo_matrix((offset_data, (valid_rows, valid_cols)), 
                                       shape=(n_unknowns, n_unknowns), dtype=np.float32)
            
            # Add to matrix incrementally
            A = A + offset_coo
            
            # Clear temporary arrays to free memory
            del offset_data, offset_coo, valid_rows, valid_cols
            
            # Boundary Conditions
            # If neighbor is OUTSIDE (Epi), add to RHS
            is_epi = (neighbor_labels == self.STATE_OUTSIDE)
            rhs[is_epi] += 1.0
        
        self.timing_stats['build_matrix_arrays'] = time.time() - t_start
        print(f"  ✓ Build matrix incrementally: {self.timing_stats['build_matrix_arrays']:.3f}s")
        
        t_start = time.time()
        # Convert COO to CSR for efficient solving
        A = A.tocsr()
        nnz = A.nnz
        self.timing_stats['create_sparse_matrix'] = time.time() - t_start
        print(f"  ✓ Convert to CSR format (float32): {self.timing_stats['create_sparse_matrix']:.3f}s")
        print(f"    Matrix: {n_unknowns:,} x {n_unknowns:,}, {nnz:,} non-zeros")
        print(f"    Memory: ~{(nnz * 4 + n_unknowns * 8) / 1024**2:.1f} MB (matrix data)")
        
        t_start = time.time()
        if CUPY_AVAILABLE:
            # Use GPU solver for maximum performance
            try:
                print(f"  Transferring to GPU...")
                t_transfer = time.time()
                
                # Already float32, no conversion needed
                A_f32 = A
                rhs_f32 = rhs
                
                # Transfer to GPU
                A_gpu = cpsp.csr_matrix(A_f32)
                rhs_gpu = cp.array(rhs_f32)
                
                transfer_time = time.time() - t_transfer
                print(f"    Transfer to GPU: {transfer_time:.3f}s")
                
                # Solve on GPU using CG with diagonal preconditioner
                t_solve = time.time()
                
                # Create diagonal preconditioner (Jacobi)
                diag = cp.array(A_f32.diagonal())
                diag[diag == 0] = 1.0
                M_inv = cpsp.diags(1.0 / diag, format='csr')
                
                # Solve with preconditioned CG
                x_gpu, info = cpsla.cg(A_gpu, rhs_gpu, M=M_inv, tol=1e-5, maxiter=500)
                
                if info != 0:
                    print(f"  Warning: CG did not converge (info={info}), falling back to CPU")
                    raise RuntimeError("CG failed")
                
                solve_time = time.time() - t_solve
                print(f"    GPU solve: {solve_time:.3f}s (CG iterations: {info if info > 0 else 'converged'})")
                
                # Transfer back to CPU
                t_back = time.time()
                x = cp.asnumpy(x_gpu).astype(np.float64)
                back_time = time.time() - t_back
                print(f"    Transfer from GPU: {back_time:.3f}s")
                
                self.timing_stats['solve_method'] = 'CuPy-GPU'
                self.timing_stats['gpu_transfer_to'] = transfer_time
                self.timing_stats['gpu_solve'] = solve_time
                self.timing_stats['gpu_transfer_from'] = back_time
                print(f"  ✓ Sparse solve (GPU): {time.time() - t_start:.3f}s")
                
            except Exception as e:
                print(f"  GPU solve failed ({e}), falling back to CPU...")
                # Fall back to CPU solver
                if PYAMG_AVAILABLE:
                    ml = pyamg.smoothed_aggregation_solver(A, max_levels=10, max_coarse=500)
                    x = ml.solve(rhs, tol=1e-6, accel='cg')
                    self.timing_stats['solve_method'] = 'PyAMG (GPU fallback)'
                    print(f"  ✓ Sparse solve (PyAMG): {time.time() - t_start:.3f}s")
                else:
                    x = spsolve(A, rhs)
                    self.timing_stats['solve_method'] = 'spsolve (GPU fallback)'
                    print(f"  ✓ Sparse solve (spsolve): {time.time() - t_start:.3f}s")
        
        elif PYAMG_AVAILABLE:
            # Use PyAMG for faster solving (optimal for Laplace equations)
            # PyAMG works with float32 matrices
            ml = pyamg.smoothed_aggregation_solver(A, max_levels=10, max_coarse=500)
            x = ml.solve(rhs, tol=1e-5, accel='cg')
            self.timing_stats['solve_method'] = 'PyAMG (float32)'
            print(f"  ✓ Sparse solve (PyAMG float32): {time.time() - t_start:.3f}s")
        else:
            # Fallback to direct solver
            x = spsolve(A, rhs)
            self.timing_stats['solve_method'] = 'spsolve (float32)'
            print(f"  ✓ Sparse solve (spsolve float32): {time.time() - t_start:.3f}s")
        
        self.timing_stats['spsolve'] = time.time() - t_start
        
        # Map back to full grid
        t_start = time.time()
        self.potential_field = np.zeros(n_voxels, dtype=np.float32)
        # Set boundaries
        self.potential_field[labels_flat == self.STATE_OUTSIDE] = 1.0
        self.potential_field[labels_flat == self.STATE_BLOOD] = 0.0
        # Set calculated wall (x is already float32 or converted from float64)
        self.potential_field[unknown_indices] = x.astype(np.float32) if x.dtype != np.float32 else x
        
        # Reshape to 3D
        self.potential_field = self.potential_field.reshape(self.dims)
        self.timing_stats['map_solution'] = time.time() - t_start
        print(f"  ✓ Map solution to grid: {self.timing_stats['map_solution']:.3f}s")
        
        self.timing_stats['solve_laplace_total'] = time.time() - t_start_total
        print(f"\n{'='*70}")
        print(f"LAPLACE SOLVE COMPLETE: {self.timing_stats['solve_laplace_total']:.3f}s")
        print(f"{'='*70}")

    def calculate_thickness(self):
        """
        Compute gradients and integrate streamlines from Endo to Epi.
        """
        t_start_total = time.time()
        print(f"\n{'='*70}")
        print(f"CALCULATING THICKNESS MAP")
        print(f"{'='*70}")
        
        # 1. Compute Gradients on coarser grid for memory efficiency
        t_start = time.time()
        
        # Determine gradient resolution: use coarser grid if base resolution is fine
        # For resolutions <= 0.15mm, use 2x downsampling to save memory
        gradient_downsample = 1
        if self.resolution <= 0.15:
            gradient_downsample = 2
            print(f"  Using {gradient_downsample}x downsampled gradient field to reduce memory")
        
        if gradient_downsample > 1:
            # Downsample potential field for gradient computation
            from scipy.ndimage import zoom
            downsample_factor = 1.0 / gradient_downsample
            
            print(f"  Downsampling potential field from {self.dims} to coarser grid...")
            t_downsample = time.time()
            potential_coarse = zoom(self.potential_field, downsample_factor, order=1)
            coarse_dims = potential_coarse.shape
            coarse_spacing = [s * gradient_downsample for s in self.spacing]
            print(f"    Coarse grid: {coarse_dims} ({coarse_dims[0]*coarse_dims[1]*coarse_dims[2]:,} voxels)")
            print(f"    Downsample time: {time.time() - t_downsample:.3f}s")
            
            # Compute gradients on coarser grid
            print(f"  Computing gradients on coarse grid...")
            t_grad = time.time()
            grad = np.gradient(potential_coarse, coarse_spacing[0], coarse_spacing[1], coarse_spacing[2])
            grad_x, grad_y, grad_z = grad[0], grad[1], grad[2]
            print(f"    Gradient computation time: {time.time() - t_grad:.3f}s")
            
            # Normalize to get direction
            norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            norm[norm == 0] = 1.0
            
            dir_x = grad_x / norm
            dir_y = grad_y / norm
            dir_z = grad_z / norm
            
            # Update dimensions and spacing for interpolators
            dims_for_interp = coarse_dims
            spacing_for_interp = coarse_spacing
            potential_for_interp = potential_coarse
            
            # Free memory
            del grad, grad_x, grad_y, grad_z, norm
        else:
            # Use full resolution gradient
            grad = np.gradient(self.potential_field, self.spacing[0], self.spacing[1], self.spacing[2])
            grad_x, grad_y, grad_z = grad[0], grad[1], grad[2]
            
            # Normalize to get direction
            norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            norm[norm == 0] = 1.0
            
            dir_x = grad_x / norm
            dir_y = grad_y / norm
            dir_z = grad_z / norm
            
            dims_for_interp = self.dims
            spacing_for_interp = self.spacing
            potential_for_interp = self.potential_field
        
        self.timing_stats['compute_gradients'] = time.time() - t_start
        print(f"  ✓ Compute gradients: {self.timing_stats['compute_gradients']:.3f}s")
        
        # Interpolators for direction field
        # Prepare grid coordinates (using coarse grid if downsampled)
        t_start = time.time()
        x = np.linspace(self.origin[0], self.origin[0] + (dims_for_interp[0]-1)*spacing_for_interp[0], dims_for_interp[0])
        y = np.linspace(self.origin[1], self.origin[1] + (dims_for_interp[1]-1)*spacing_for_interp[1], dims_for_interp[1])
        z = np.linspace(self.origin[2], self.origin[2] + (dims_for_interp[2]-1)*spacing_for_interp[2], dims_for_interp[2])
        
        interp_dx = RegularGridInterpolator((x, y, z), dir_x, bounds_error=False, fill_value=0)
        interp_dy = RegularGridInterpolator((x, y, z), dir_y, bounds_error=False, fill_value=0)
        interp_dz = RegularGridInterpolator((x, y, z), dir_z, bounds_error=False, fill_value=0)
        interp_pot = RegularGridInterpolator((x, y, z), potential_for_interp, bounds_error=False, fill_value=1)
        
        self.timing_stats['create_interpolators'] = time.time() - t_start
        print(f"  ✓ Create interpolators: {self.timing_stats['create_interpolators']:.3f}s")

        # 2. Iterate over Endo Mesh Points
        t_start = time.time()
        endo_points = vtk_to_numpy(self.endo_poly.GetPoints().GetData())
        n_points = len(endo_points)
        thicknesses = np.zeros(n_points)
        
        print(f"Tracing streamlines for {n_points} points...")
        
        # Integration parameters
        step_size = self.resolution * 0.5 
        max_steps = int(20.0 / step_size) # Max 20mm thickness
        
        print(f"  Processing {n_points:,} points with step_size={step_size:.4f}mm, max_steps={max_steps}")
        
        t_streamline_start = time.time()
        
        if NUMBA_AVAILABLE:
            # Use Numba-accelerated streamline tracing
            print(f"  Using Numba JIT compilation (first run may be slow due to compilation)")
            origin_array = np.array(self.origin, dtype=np.float64)
            spacing_array = np.array(spacing_for_interp, dtype=np.float64)
            dims_array = np.array(dims_for_interp, dtype=np.int64)
            
            for i in range(n_points):
                thicknesses[i] = trace_streamline_numba(
                    endo_points[i], dir_x, dir_y, dir_z, potential_for_interp,
                    origin_array, spacing_array, dims_array, step_size, max_steps
                )
                
                if i % 5000 == 0 and i > 0:
                    elapsed = time.time() - t_streamline_start
                    rate = i / elapsed
                    eta = (n_points - i) / rate
                    print(f"  Progress: {i:,}/{n_points:,} ({100*i/n_points:.1f}%) - Rate: {rate:.0f} pts/s - ETA: {eta:.1f}s")
        else:
            # Fallback to original implementation
            print(f"  Using standard Python (install numba for 5-10x speedup)")
            for i, start_pt in enumerate(endo_points):
                curr_pos = start_pt.copy()
                length = 0.0
                
                for step in range(max_steps):
                    # Sample direction
                    d = np.array([
                        interp_dx([curr_pos])[0],
                        interp_dy([curr_pos])[0],
                        interp_dz([curr_pos])[0]
                    ])
                    
                    # Check for zero gradient
                    d_norm = np.linalg.norm(d)
                    if d_norm < 0.001:
                        break
                        
                    # Take step
                    step_vec = d * step_size
                    curr_pos += step_vec
                    length += step_size
                    
                    # Check potential
                    pot = interp_pot([curr_pos])[0]
                    if pot >= 0.95:
                        break
                
                thicknesses[i] = length
                
                if i % 5000 == 0 and i > 0:
                    elapsed = time.time() - t_streamline_start
                    rate = i / elapsed
                    eta = (n_points - i) / rate
                    print(f"  Progress: {i:,}/{n_points:,} ({100*i/n_points:.1f}%) - Rate: {rate:.0f} pts/s - ETA: {eta:.1f}s")
        
        self.timing_stats['streamline_tracing'] = time.time() - t_streamline_start
        print(f"  ✓ Streamline tracing: {self.timing_stats['streamline_tracing']:.3f}s ({n_points/self.timing_stats['streamline_tracing']:.1f} pts/s)")
        
        # 3. Add thickness to polydata
        t_start = time.time()
        arr = numpy_to_vtk(thicknesses)
        arr.SetName("LaplaceThickness")
        
        pd = self.endo_poly.GetPointData()
        pd.AddArray(arr)
        pd.SetActiveScalars("LaplaceThickness")
        
        self.endo_poly.GetPointData().SetActiveScalars("LaplaceThickness")
        
        self.timing_stats['add_to_polydata'] = time.time() - t_start
        print(f"  ✓ Add to polydata: {self.timing_stats['add_to_polydata']:.3f}s")
        
        self.timing_stats['calculate_thickness_total'] = time.time() - t_start_total
        print(f"\n{'='*70}")
        print(f"THICKNESS CALCULATION COMPLETE: {self.timing_stats['calculate_thickness_total']:.3f}s")
        print(f"{'='*70}")
        

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
        t_start_total = time.time()
        print(f"\n{'#'*70}")
        print(f"#  FAST LAPLACE WALL THICKNESS CALCULATION")
        print(f"#  Resolution: {self.resolution}mm")
        print(f"#  Endo points: {self.endo_poly.GetNumberOfPoints():,}")
        print(f"#  Epi points: {self.epi_poly.GetNumberOfPoints():,}")
        print(f"{'#'*70}")
        
        # Execute algorithm
        count = self.voxelize_domain()
        
        if count > 0:
            self.solve_laplace()
            self.calculate_thickness()
            self.save_output(vtk_filename)
            self.analyze_regions(csv_filename)
            
            total_time = time.time() - t_start_total
            self.timing_stats['total_execution'] = total_time
            
            # Print summary
            print(f"\n{'#'*70}")
            print(f"#  EXECUTION COMPLETE")
            print(f"{'#'*70}")
            print(f"\n** TIMING SUMMARY **")
            print(f"  Voxelization:       {self.timing_stats.get('voxelize_total', 0):>8.3f}s ({100*self.timing_stats.get('voxelize_total', 0)/total_time:>5.1f}%)")
            print(f"    - Endo voxel:     {self.timing_stats.get('endo_voxelization', 0):>8.3f}s")
            print(f"    - Epi voxel:      {self.timing_stats.get('epi_voxelization', 0):>8.3f}s")
            print(f"  Laplace Solve:      {self.timing_stats.get('solve_laplace_total', 0):>8.3f}s ({100*self.timing_stats.get('solve_laplace_total', 0)/total_time:>5.1f}%)")
            print(f"    - Matrix build:   {self.timing_stats.get('build_matrix_arrays', 0):>8.3f}s")
            print(f"    - Sparse matrix:  {self.timing_stats.get('create_sparse_matrix', 0):>8.3f}s")
            print(f"    - Solver ({self.timing_stats.get('solve_method', 'unknown')}): {self.timing_stats.get('spsolve', 0):>8.3f}s")
            print(f"  Thickness Calc:     {self.timing_stats.get('calculate_thickness_total', 0):>8.3f}s ({100*self.timing_stats.get('calculate_thickness_total', 0)/total_time:>5.1f}%)")
            print(f"    - Gradients:      {self.timing_stats.get('compute_gradients', 0):>8.3f}s")
            print(f"    - Interpolators:  {self.timing_stats.get('create_interpolators', 0):>8.3f}s")
            print(f"    - Streamlines:    {self.timing_stats.get('streamline_tracing', 0):>8.3f}s")
            print(f"  {'─'*40}")
            print(f"  TOTAL:              {total_time:>8.3f}s")
            print(f"\n{'#'*70}\n")
            
            return True
        else:
            print("Error: Voxelization failed or empty wall domain.")
            return False
