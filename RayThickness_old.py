import vtk
import numpy as np
import time
import csv
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


class RayWallThickness:
    """Ray casting wall thickness plugin."""
    
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
    
    def __init__(
        self,
        endo_poly,
        epi_poly,
        grid_bounds=None,
        resolution=1.0,
        max_thickness_mm=10.0,
        min_thickness_mm=0.1,
        outlier_std_threshold=3.0,
    ):
        self.endo_poly = endo_poly
        self.epi_poly = epi_poly
        self.grid_bounds = grid_bounds
        self.resolution = resolution
        self.max_thickness_mm = max_thickness_mm
        self.min_thickness_mm = min_thickness_mm
        self.outlier_std_threshold = outlier_std_threshold
        
        self.thickness_per_point = None
        self.thickness_by_region = None
        self.discard_reasons = None
    
    def _get_regions_from_endo(self):
        regions_arr = self.endo_poly.GetPointData().GetArray("Regions")
        if not regions_arr:
            raise ValueError("'Regions' array not found on endocardium mesh.")
        return vtk_to_numpy(regions_arr)
    
    def _compute_point_normals(self):
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(self.endo_poly)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.AutoOrientNormalsOn()
        normals_filter.ConsistencyOn()
        normals_filter.SplittingOff()
        normals_filter.Update()
        
        normals_pd = normals_filter.GetOutput()
        normals_arr = normals_pd.GetPointData().GetArray("Normals")
        if not normals_arr:
            raise ValueError("Failed to compute point normals for endocardium mesh.")
        return vtk_to_numpy(normals_arr)
    
    def calculate_thickness(self):
        """
        Calculate wall thickness using ray casting along vertex normals (SLOW but ACCURATE).
        
        This method casts a ray from each interior vertex along its normal direction
        and finds the intersection with the exterior mesh. This provides true wall
        thickness measurements rather than nearest-point approximations.
        """
        print("\n" + "="*60)
        print("  CALCULATING WALL THICKNESS (Ray Casting - ACCURATE)")
        print("="*60 + "\n")
        
        print("⚠ This method is slower but more accurate than the fast method.")
        print("  It casts rays along vertex normals to find true wall thickness.\n")
        
        interior_points = vtk_to_numpy(self.endo_poly.GetPoints().GetData())
        interior_regions = self._get_regions_from_endo()
        exterior_points = vtk_to_numpy(self.epi_poly.GetPoints().GetData())
    
    print(f"Exterior surface: {len(exterior_points)} vertices")
    print(f"Interior mesh: {len(interior_points)} vertices")
    
    # === ALIGNMENT CHECK ===
    print("\nChecking mesh alignment...")
    interior_centroid = np.mean(interior_points, axis=0)
    exterior_centroid = np.mean(exterior_points, axis=0)
    centroid_offset = np.linalg.norm(exterior_centroid - interior_centroid)
    
    print(f"  Interior centroid: ({interior_centroid[0]:.2f}, {interior_centroid[1]:.2f}, {interior_centroid[2]:.2f})")
    print(f"  Exterior centroid: ({exterior_centroid[0]:.2f}, {exterior_centroid[1]:.2f}, {exterior_centroid[2]:.2f})")
    print(f"  Centroid offset: {centroid_offset:.2f} mm")
    
    if centroid_offset > 5.0:
        print(f"  ⚠ WARNING: Mesh centroids differ by {centroid_offset:.2f}mm!")
        print(f"    This may indicate meshes are not aligned.")
    else:
        print(f"  ✓ Mesh alignment appears OK (offset < 5mm)")
    
        print(f"\nFiltering parameters:")
        print(f"  Min thickness: {self.min_thickness_mm} mm")
        print(f"  Max thickness: {self.max_thickness_mm} mm")
        print(f"  Outlier threshold: {self.outlier_std_threshold} standard deviations")
        
        # Compute normals for interior mesh vertices
        print("\nComputing interior vertex normals...")
        t1 = time.time()
        normals = self._compute_point_normals()
        t2 = time.time()
        print(f"  Normal computation time: {(t2-t1):.2f}s")
        
        # Build OBBTree for ray casting on exterior mesh
        print("\nBuilding OBBTree for ray casting...")
        t1 = time.time()
        obb_tree = vtk.vtkOBBTree()
        obb_tree.SetDataSet(self.epi_poly)
        obb_tree.BuildLocator()
        t2 = time.time()
        print(f"  OBBTree build time: {(t2-t1):.2f}s")
    
    # Determine normal direction (inward vs outward)
    print("\nDetermining normal direction...")
    print("=" * 60)
    
    # First, verify mesh relationship by checking actual surface-to-surface distance
    print("\n1. MESH DISTANCE VERIFICATION:")
    from scipy.spatial import KDTree
    kdtree_check = KDTree(exterior_points)
    sample_distances = []
    sample_indices = np.random.choice(len(interior_points), min(1000, len(interior_points)), replace=False)
    
    for idx in sample_indices:
        if interior_regions[idx] <= 0:
            continue
        dist, _ = kdtree_check.query(interior_points[idx])
        sample_distances.append(dist)
    
    if sample_distances:
        min_dist = np.min(sample_distances)
        max_dist = np.max(sample_distances)
        mean_dist = np.mean(sample_distances)
        median_dist = np.median(sample_distances)
        
        print(f"  Sample surface-to-surface distances (n={len(sample_distances)}):")
        print(f"    Min:    {min_dist:.3f} mm")
        print(f"    Max:    {max_dist:.3f} mm")
        print(f"    Mean:   {mean_dist:.3f} mm")
        print(f"    Median: {median_dist:.3f} mm")
        
        if max_dist > max_thickness_mm:
            print(f"  ⚠ WARNING: Max distance ({max_dist:.1f}mm) exceeds ray length ({max_thickness_mm}mm)!")
            print(f"    Rays may be too short to reach exterior mesh.")
        
        if min_dist < 0.5:
            print(f"  ⚠ WARNING: Min distance ({min_dist:.3f}mm) is very small!")
            print(f"    Meshes may be overlapping or very close.")
    
    # Method 1: Geometric heuristic - exterior should be in direction of normals
    # Sample vertices and check if normal points toward exterior centroid
    print("\n2. GEOMETRIC TEST (normal vs exterior centroid):")
    geometric_vote_outward = 0
    geometric_vote_inward = 0
    
    sample_size = min(1000, len(interior_points))
    sampled_indices = np.random.choice(len(interior_points), sample_size, replace=False)
    
    for i in sampled_indices:
        region_id = interior_regions[i]
        if region_id <= 0:
            continue
        
        pt = interior_points[i]
        normal = normals[i]
        
        # Vector from this point to exterior centroid
        to_exterior_centroid = exterior_centroid - pt
        
        # Check if normal points toward exterior centroid
        dot = np.dot(normal, to_exterior_centroid)
        
        if dot > 0:
            geometric_vote_outward += 1
        else:
            geometric_vote_inward += 1
    
    print(f"    Normals pointing toward exterior centroid: {geometric_vote_outward}")
    print(f"    Normals pointing away from exterior:      {geometric_vote_inward}")
    
    # Method 2: Check if normal points away from interior centroid
    print("\n3. INTERIOR CENTROID TEST (normal vs interior centroid):")
    interior_test_outward = 0
    interior_test_inward = 0
    
    for i in sampled_indices:
        region_id = interior_regions[i]
        if region_id <= 0:
            continue
        
        pt = interior_points[i]
        normal = normals[i]
        
        # Vector from interior centroid to this point (should align with outward normal)
        from_interior_centroid = pt - interior_centroid
        from_interior_centroid_norm = from_interior_centroid / (np.linalg.norm(from_interior_centroid) + 1e-10)
        
        dot = np.dot(normal, from_interior_centroid_norm)
        
        if dot > 0:
            interior_test_outward += 1
        else:
            interior_test_inward += 1
        
        print(f"    Normals pointing away from interior center: {interior_test_outward}")
        print(f"    Normals pointing toward interior center:   {interior_test_inward}")
        
        # Method 3: Ray casting test - which direction gets more hits
        print("\n4. RAY CASTING TEST (actual intersection test):")
        sample_count = 0
        outward_count = 0
        inward_count = 0
        outward_distances = []
        inward_distances = []
        
        test_ray_length = max(30.0, self.max_thickness_mm * 2)  # Use longer rays for testing
        
        for i in sampled_indices[:500]:  # Test with 500 rays
            region_id = interior_regions[i]
            if region_id <= 0:
                continue
            
            pt = interior_points[i]
            normal = normals[i]
            
            # Test ray in NORMAL direction (outward if normals point out)
            ray_start_out = pt + normal * 0.01
            ray_end_outward = pt + normal * test_ray_length
            
            # Test ray in OPPOSITE direction (outward if normals point in)
            ray_start_in = pt - normal * 0.01
            ray_end_inward = pt - normal * test_ray_length
            
            # Check intersection in both directions
            points_outward = vtk.vtkPoints()
            points_inward = vtk.vtkPoints()
            
            hit_outward = obb_tree.IntersectWithLine(ray_start_out, ray_end_outward, points_outward, None)
            hit_inward = obb_tree.IntersectWithLine(ray_start_in, ray_end_inward, points_inward, None)
            
            if hit_outward and points_outward.GetNumberOfPoints() > 0:
                outward_count += 1
                # Get first intersection distance
                first_pt = np.array(points_outward.GetPoint(0))
                dist = np.linalg.norm(first_pt - pt)
                outward_distances.append(dist)
                
            if hit_inward and points_inward.GetNumberOfPoints() > 0:
                inward_count += 1
                first_pt = np.array(points_inward.GetPoint(0))
                dist = np.linalg.norm(first_pt - pt)
                inward_distances.append(dist)
            
            sample_count += 1
        
        print(f"    Rays in NORMAL direction:   {outward_count} hits")
            if outward_distances:
                print(f"      Hit distances: min={np.min(outward_distances):.2f}, max={np.max(outward_distances):.2f}, mean={np.mean(outward_distances):.2f} mm")
        print(f"    Rays in OPPOSITE direction: {inward_count} hits")
        if inward_distances:
            print(f"      Hit distances: min={np.min(inward_distances):.2f}, max={np.max(inward_distances):.2f}, mean={np.mean(inward_distances):.2f} mm")
        
        # Decision logic
        print("\n5. DECISION:")
        print("=" * 60)
        
        # Strong preference for ray casting test since it's most direct
        if outward_count > inward_count * 2:
            print(f"  → RAY TEST is decisive: Normals point OUTWARD")
            print(f"     Using normal direction AS-IS")
            normal_sign = 1.0
        elif inward_count > outward_count * 2:
            print(f"  → RAY TEST is decisive: Normals point INWARD")
            print(f"     Will NEGATE normals for ray casting")
            normal_sign = -1.0
        elif outward_count > inward_count:
            print(f"  → RAY TEST favors OUTWARD (but not decisive)")
            print(f"     Using normal direction AS-IS")
            normal_sign = 1.0
        elif inward_count > outward_count:
            print(f"  → RAY TEST favors INWARD (but not decisive)")
            print(f"     Will NEGATE normals for ray casting")
            normal_sign = -1.0
        else:
            print(f"  ⚠ RAY TEST is INCONCLUSIVE!")
            # Fall back to geometric tests
            if interior_test_outward > interior_test_inward:
                print(f"     Falling back to INTERIOR CENTROID test: Using normals AS-IS")
                normal_sign = 1.0
            else:
                print(f"     Falling back to INTERIOR CENTROID test: Will NEGATE normals")
                normal_sign = -1.0
        
        print("=" * 60)
        
        # Dictionary to store thickness measurements
            thickness_by_region = {}
        discard_reasons = {}
        for region_id in range(17):
            thickness_by_region[region_id] = []
            discard_reasons[region_id] = {
                'too_close': 0,
                'too_far': 0,
                'no_intersection': 0,
                'outlier': 0,
            }
        
        # Cast rays for all interior vertices
        print(f"\nCasting rays for {len(interior_points)} vertices...")
        print(f"  This may take several minutes...")
        print(f"  Using normal_sign = {normal_sign} ({'NEGATED' if normal_sign < 0 else 'AS-IS'})")
        print(f"  Ray length = {self.max_thickness_mm} mm")
        
        # Debug: Show details for first few vertices
        print("\n  DEBUG: Sample ray details for first 5 valid vertices:")
        debug_count = 0
            if debug_count >= 5:
                break
            region_id = interior_regions[i]
            if region_id <= 0:
                continue
            
            pt = interior_points[i]
            normal = normals[i] * normal_sign
            ray_start = pt + normal * 0.01
            ray_end = pt + normal * self.max_thickness_mm
            
            print(f"    Vertex {i} (region {region_id}):")
            print(f"      Position: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
            print(f"      Normal (signed): ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
            print(f"      Ray start: ({ray_start[0]:.2f}, {ray_start[1]:.2f}, {ray_start[2]:.2f})")
            print(f"      Ray end:   ({ray_end[0]:.2f}, {ray_end[1]:.2f}, {ray_end[2]:.2f})")
            
            # Test this ray
            intersection_points = vtk.vtkPoints()
            result = obb_tree.IntersectWithLine(ray_start, ray_end, intersection_points, None)
            print(f"      Intersections: {intersection_points.GetNumberOfPoints()}")
            
            debug_count += 1
        
        print()
        t1 = time.time()
        
        progress_interval = len(interior_points) // 10
        processed = 0
            region_id = interior_regions[i]
            if region_id <= 0:
                continue
            
            pt = interior_points[i]
            normal = normals[i] * normal_sign
            
            # Cast ray from slightly offset position to avoid self-intersection issues
            # Start 0.01mm along the normal to ensure clean intersection detection
            ray_start = pt + normal * 0.01
            ray_end = pt + normal * self.max_thickness_mm
            
            intersection_points = vtk.vtkPoints()
            result = obb_tree.IntersectWithLine(ray_start, ray_end, intersection_points, None)
            
            if result == 0 or intersection_points.GetNumberOfPoints() == 0:
                discard_reasons[region_id]['no_intersection'] += 1
                continue
            
            # Find FIRST intersection along ray direction (smallest parametric t)
            # Don't use closest distance - use parametric distance along ray
            min_t = float('inf')
            first_intersection = None
            ray_direction = normal
            
            for j in range(intersection_points.GetNumberOfPoints()):
                intersection_pt = np.array(intersection_points.GetPoint(j))
                # Compute parametric distance t along ray: intersection = start + t * direction
                to_intersection = intersection_pt - pt
                t = np.dot(to_intersection, ray_direction)
                
                if t > 0 and t < min_t:  # Only consider intersections ahead of start point
                    min_t = t
                    first_intersection = intersection_pt
            
            if first_intersection is None:
                discard_reasons[region_id]['no_intersection'] += 1
                continue
            
            distance = min_t  # Parametric distance IS the actual distance since ray_direction is normalized
            
            # Apply filters
            if distance < self.min_thickness_mm:
                discard_reasons[region_id]['too_close'] += 1
                continue
            
            if distance > self.max_thickness_mm:
                discard_reasons[region_id]['too_far'] += 1
                continue
            
            thickness_by_region[region_id].append(distance)
            
            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                elapsed = time.time() - t1
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (len(interior_points) - processed) / rate if rate > 0 else 0
                print(f"    Processed {processed} vertices ({100*processed/len(interior_points):.0f}%), "
                      f"~{remaining:.0f}s remaining")
        
        t2 = time.time()
        total_time = t2 - t1
        print(f"\n  Ray casting completed in {total_time:.1f}s")
        print(f"  Rate: {len(interior_points) / total_time:.0f} vertices/sec")
        
        # Report discard statistics
        print(f"\n  Discard statistics by reason:")
        total_no_intersection = sum(d['no_intersection'] for d in discard_reasons.values())
        total_too_close = sum(d['too_close'] for d in discard_reasons.values())
        total_too_far = sum(d['too_far'] for d in discard_reasons.values())
        print(f"    No intersection: {total_no_intersection}")
        print(f"    Too close (<{self.min_thickness_mm}mm): {total_too_close}")
        print(f"    Too far (>{self.max_thickness_mm}mm): {total_too_far}")
        
        # === OUTLIER DETECTION PER REGION ===
        print(f"\nApplying outlier detection (>{self.outlier_std_threshold}σ)...")
        outliers_removed = 0
        
        for region_id in range(1, 17):
            if len(thickness_by_region[region_id]) < 10:
                continue
            
            thicknesses = np.array(thickness_by_region[region_id])
            mean_t = np.mean(thicknesses)
            std_t = np.std(thicknesses)
            
            if std_t < 1e-6:
                continue
            
            lower_bound = mean_t - self.outlier_std_threshold * std_t
            upper_bound = mean_t + self.outlier_std_threshold * std_t
            
            filtered = thicknesses[(thicknesses >= lower_bound) & (thicknesses <= upper_bound)]
            removed = len(thicknesses) - len(filtered)
            
            if removed > 0:
                outliers_removed += removed
                thickness_by_region[region_id] = filtered.tolist()
                discard_reasons[region_id]['outlier'] += removed
        
        if outliers_removed > 0:
            print(f"  Removed {outliers_removed} statistical outliers across all regions")
        else:
            print(f"  No statistical outliers detected")
        
        def total_discards(region_id):
            return sum(discard_reasons[region_id].values())
        
        # Store results for later use
        self.thickness_by_region = thickness_by_region
        self.discard_reasons = discard_reasons
        
        # Convert to per-point thickness array
        thickness_per_point = np.full(len(interior_points), -1.0)
        # Note: We can't easily map back from region lists to individual points
        # This is a limitation of the ray casting approach
        self.thickness_per_point = thickness_per_point
        
        # Print results
        region_names = [
            'Background', 'RSPV', 'LSPV', 'RIPV', 'LIPV', 'MA', 'LAA',
            'Posterior', 'Roof', 'Inferior', 'Lateral',
            'Septal', 'Anterior', 'RSPV_Ostium', 'LSPV_Ostium',
            'RIPV_Ostium', 'LIPV_Ostium'
        ]
        
        results_data = []
    
    # Print WALL REGIONS
    print("\n" + "="*120)
    print("  WALL THICKNESS RESULTS - PRIMARY WALL REGIONS (Ray Casting)")
    print("="*120)
    print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'NoHit':<10} {'TooClose':<10} {'TooFar':<10} {'Outlier':<10}")
    print("-" * 120)
    
    for region_id in [7, 8, 9, 10, 11, 12]:
        if region_id in thickness_by_region and len(thickness_by_region[region_id]) > 0:
            thicknesses = thickness_by_region[region_id]
            avg_thickness = np.mean(thicknesses)
            std_thickness = np.std(thicknesses)
            num_valid = len(thicknesses)
            name = region_names[region_id]
            dr = discard_reasons[region_id]
            
            print(f"{region_id:<8} {name:<18} {avg_thickness:<12.3f} {std_thickness:<12.3f} {num_valid:<10} {dr['no_intersection']:<10} {dr['too_close']:<10} {dr['too_far']:<10} {dr['outlier']:<10}")
            
            results_data.append({
                'Region_ID': region_id,
                'Region_Name': name,
                'Category': 'Wall',
                'Avg_Thickness_mm': round(avg_thickness, 4),
                'Std_Dev_mm': round(std_thickness, 4),
                'Valid_Vertices': num_valid,
                'Discard_NoIntersection': dr['no_intersection'],
                'Discard_TooClose': dr['too_close'],
                'Discard_TooFar': dr['too_far'],
                'Discard_Outlier': dr['outlier'],
                'Total_Vertices': num_valid + total_discards(region_id)
            })
    
    print("-" * 120)
    
    # Print OSTIUM REGIONS
    print("\n  OSTIUM REGIONS (Transition Zones)")
    print("-" * 120)
    print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'NoHit':<10} {'TooClose':<10} {'TooFar':<10} {'Outlier':<10}")
    print("-" * 120)
    
    for region_id in [13, 14, 15, 16]:
        if region_id in thickness_by_region and len(thickness_by_region[region_id]) > 0:
            thicknesses = thickness_by_region[region_id]
            avg_thickness = np.mean(thicknesses)
            std_thickness = np.std(thicknesses)
            num_valid = len(thicknesses)
            name = region_names[region_id]
            dr = discard_reasons[region_id]
            
            print(f"{region_id:<8} {name:<18} {avg_thickness:<12.3f} {std_thickness:<12.3f} {num_valid:<10} {dr['no_intersection']:<10} {dr['too_close']:<10} {dr['too_far']:<10} {dr['outlier']:<10}")
            
            results_data.append({
                'Region_ID': region_id,
                'Region_Name': name,
                'Category': 'Ostium',
                'Avg_Thickness_mm': round(avg_thickness, 4),
                'Std_Dev_mm': round(std_thickness, 4),
                'Valid_Vertices': num_valid,
                'Discard_NoIntersection': dr['no_intersection'],
                'Discard_TooClose': dr['too_close'],
                'Discard_TooFar': dr['too_far'],
                'Discard_Outlier': dr['outlier'],
                'Total_Vertices': num_valid + total_discards(region_id)
            })
    
    print("-" * 120)
    
    # Print PV and OTHER REGIONS
    print("\n  PV & SPECIAL STRUCTURES")
    print("-" * 120)
    print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'NoHit':<10} {'TooClose':<10} {'TooFar':<10} {'Outlier':<10}")
    print("-" * 120)
    
    for region_id in [1, 2, 3, 4, 5, 6]:
        if region_id in thickness_by_region:
            thicknesses = thickness_by_region[region_id]
            name = region_names[region_id]
            dr = discard_reasons[region_id]
            
            if len(thicknesses) > 0:
                avg_thickness = np.mean(thicknesses)
                std_thickness = np.std(thicknesses)
                num_valid = len(thicknesses)
                print(f"{region_id:<8} {name:<18} {avg_thickness:<12.3f} {std_thickness:<12.3f} {num_valid:<10} {dr['no_intersection']:<10} {dr['too_close']:<10} {dr['too_far']:<10} {dr['outlier']:<10}")
                
                results_data.append({
                    'Region_ID': region_id,
                    'Region_Name': name,
                    'Category': 'PV/Special',
                    'Avg_Thickness_mm': round(avg_thickness, 4),
                    'Std_Dev_mm': round(std_thickness, 4),
                    'Valid_Vertices': num_valid,
                    'Discard_NoIntersection': dr['no_intersection'],
                    'Discard_TooClose': dr['too_close'],
                    'Discard_TooFar': dr['too_far'],
                    'Discard_Outlier': dr['outlier'],
                    'Total_Vertices': num_valid + total_discards(region_id)
                })
            elif total_discards(region_id) > 0:
                print(f"{region_id:<8} {name:<18} {'N/A':<12} {'N/A':<12} {0:<10} {dr['no_intersection']:<10} {dr['too_close']:<10} {dr['too_far']:<10} {dr['outlier']:<10}")
                
                results_data.append({
                    'Region_ID': region_id,
                    'Region_Name': name,
                    'Category': 'PV/Special',
                    'Avg_Thickness_mm': 'N/A',
                    'Std_Dev_mm': 'N/A',
                    'Valid_Vertices': 0,
                    'Discard_NoIntersection': dr['no_intersection'],
                    'Discard_TooClose': dr['too_close'],
                    'Discard_TooFar': dr['too_far'],
                    'Discard_Outlier': dr['outlier'],
                    'Total_Vertices': total_discards(region_id)
                })
        
        print("-" * 120)
        
        return results_data
    
    def add_thickness_to_mesh(self, thickness_per_point):
        """Add thickness values as point data to endocardium mesh."""
        # Ray casting approach doesn't easily map back to per-point thickness
        # This is a placeholder for compatibility
        if thickness_per_point is not None:
            arr = numpy_to_vtk(thickness_per_point)
            arr.SetName("RayThickness")
            pd = self.endo_poly.GetPointData()
            pd.AddArray(arr)
            pd.SetActiveScalars("RayThickness")
    
    def save_output(self, output_path):
        """Save modified endocardium mesh to VTK file."""
        print(f"Saving result to {output_path}")
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(self.endo_poly)
        writer.Write()
    
    def analyze_regions(self, csv_filename, results_data):
        """Save per-region thickness statistics to CSV."""
        if not results_data:
            print("No results to save.")
            return
        
        print(f"\nSaving results to CSV: {csv_filename}")
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['Region_ID', 'Region_Name', 'Category', 'Avg_Thickness_mm', 'Std_Dev_mm',
                             'Valid_Vertices', 'Discard_NoIntersection', 'Discard_TooClose',
                             'Discard_TooFar', 'Discard_Outlier', 'Total_Vertices']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_data)
            print(f"✓ Results saved to {csv_filename}")
        except Exception as e:
            print(f"✗ Failed to save CSV: {e}")
    
    def execute(self, vtk_filename, csv_filename):
        """Execute the ray casting wall thickness calculation."""
        results_data = self.calculate_thickness()
        if results_data is None or len(results_data) == 0:
            return False
        
        self.add_thickness_to_mesh(self.thickness_per_point)
        self.save_output(vtk_filename)
        self.analyze_regions(csv_filename, results_data)
        return True
