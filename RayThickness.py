import vtk
import numpy as np
import time
import csv
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


class RayWallThickness:
    """Ray casting wall thickness plugin."""
    
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
        """Calculate wall thickness using ray casting along vertex normals."""
        print("\n" + "="*60)
        print("  CALCULATING WALL THICKNESS (Ray Casting)")
        print("="*60 + "\n")
        
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
        else:
            print(f"  ✓ Mesh alignment appears OK (offset < 5mm)")
        
        # Compute normals
        print("\nComputing interior vertex normals...")
        t1 = time.time()
        normals = self._compute_point_normals()
        t2 = time.time()
        print(f"  Normal computation time: {(t2-t1):.2f}s")
        
        # Build OBBTree for ray casting
        print("\nBuilding OBBTree for ray casting...")
        t1 = time.time()
        obb_tree = vtk.vtkOBBTree()
        obb_tree.SetDataSet(self.epi_poly)
        obb_tree.BuildLocator()
        t2 = time.time()
        print(f"  OBBTree build time: {(t2-t1):.2f}s")
        
        # Determine normal direction using a more robust test
        print("\nDetermining normal direction...")
        
        # Test: Cast rays in both directions and measure which gives reasonable thicknesses
        normal_sign = 1.0
        sample_size = min(200, len(interior_points))
        sample_indices = np.random.choice(len(interior_points), sample_size, replace=False)
        
        # Filter to only wall regions (7-12) for more reliable test
        wall_sample_indices = []
        for idx in sample_indices:
            if 7 <= interior_regions[idx] <= 12:
                wall_sample_indices.append(idx)
        
        if len(wall_sample_indices) > 0:
            sample_indices = wall_sample_indices[:min(100, len(wall_sample_indices))]
        
        outward_distances = []
        inward_distances = []
        
        print("  Testing normal directions on sample wall vertices...")
        for i in sample_indices:
            region_id = interior_regions[i]
            if region_id <= 0:
                continue
            
            pt = interior_points[i]
            normal = normals[i]
            
            # Test outward ray (positive normal direction)
            ray_end_out = pt + normal * self.max_thickness_mm * 2
            points_out = vtk.vtkPoints()
            hit_out = obb_tree.IntersectWithLine(pt, ray_end_out, points_out, None)
            if hit_out > 0 and points_out.GetNumberOfPoints() > 0:
                first_intersection_out = np.array(points_out.GetPoint(0))
                dist_out = np.linalg.norm(first_intersection_out - pt)
                if 0.5 <= dist_out <= self.max_thickness_mm:  # Reasonable thickness
                    outward_distances.append(dist_out)
            
            # Test inward ray (negative normal direction)
            ray_end_in = pt - normal * self.max_thickness_mm * 2
            points_in = vtk.vtkPoints()
            hit_in = obb_tree.IntersectWithLine(pt, ray_end_in, points_in, None)
            if hit_in > 0 and points_in.GetNumberOfPoints() > 0:
                first_intersection_in = np.array(points_in.GetPoint(0))
                dist_in = np.linalg.norm(first_intersection_in - pt)
                if 0.5 <= dist_in <= self.max_thickness_mm:  # Reasonable thickness
                    inward_distances.append(dist_in)
        
        # Decide based on which direction gives more reasonable measurements
        outward_valid = len(outward_distances)
        inward_valid = len(inward_distances)
        
        print(f"  Outward direction: {outward_valid} reasonable measurements")
        if outward_valid > 0:
            print(f"    Mean distance: {np.mean(outward_distances):.2f}mm, Median: {np.median(outward_distances):.2f}mm")
        
        print(f"  Inward direction: {inward_valid} reasonable measurements")
        if inward_valid > 0:
            print(f"    Mean distance: {np.mean(inward_distances):.2f}mm, Median: {np.median(inward_distances):.2f}mm")
        
        # Use direction with more valid measurements
        if inward_valid > outward_valid * 1.5:
            normal_sign = -1.0
            print(f"  → Using INWARD direction (negating normals)")
        else:
            print(f"  → Using OUTWARD direction (normals as-is)")
        
        
        # Dictionary for measurements
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
        
        # Cast rays
        print(f"\nCasting rays for {len(interior_points)} vertices...")
        t1 = time.time()
        progress_interval = max(1, len(interior_points) // 10)
        processed = 0
        
        for i in range(len(interior_points)):
            region_id = interior_regions[i]
            if region_id <= 0:
                continue
            
            pt = interior_points[i]
            normal = normals[i] * normal_sign
            
            # Cast ray - start from point, no offset
            ray_start = pt
            ray_end = pt + normal * self.max_thickness_mm
            
            intersection_points = vtk.vtkPoints()
            result = obb_tree.IntersectWithLine(ray_start, ray_end, intersection_points, None)
            
            if result == 0 or intersection_points.GetNumberOfPoints() == 0:
                discard_reasons[region_id]['no_intersection'] += 1
                processed += 1
                continue
            
            # Find first intersection by Euclidean distance
            min_distance = float('inf')
            for j in range(intersection_points.GetNumberOfPoints()):
                intersection_pt = np.array(intersection_points.GetPoint(j))
                distance_to_intersection = np.linalg.norm(intersection_pt - pt)
                
                # Must be ahead of start point (positive distance)
                if distance_to_intersection > 1e-6 and distance_to_intersection < min_distance:
                    min_distance = distance_to_intersection
            
            if min_distance == float('inf'):
                discard_reasons[region_id]['no_intersection'] += 1
                processed += 1
                continue
            
            distance = min_distance
            
            # Apply filters
            if distance < self.min_thickness_mm:
                discard_reasons[region_id]['too_close'] += 1
            elif distance > self.max_thickness_mm:
                discard_reasons[region_id]['too_far'] += 1
            else:
                thickness_by_region[region_id].append(distance)
            
            processed += 1
            if progress_interval > 0 and processed % progress_interval == 0:
                elapsed = time.time() - t1
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (len(interior_points) - processed) / rate if rate > 0 else 0
                print(f"  Processed {processed} vertices ({100*processed/len(interior_points):.0f}%), ~{remaining:.0f}s remaining")
        
        t2 = time.time()
        print(f"\n  Ray casting completed in {t2-t1:.1f}s")
        
        # Outlier detection
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
            print(f"  Removed {outliers_removed} statistical outliers")
        
        # Store results
        self.thickness_by_region = thickness_by_region
        self.discard_reasons = discard_reasons
        self.thickness_per_point = np.full(len(interior_points), -1.0)
        
        # Format results
        region_names = [
            'Background', 'RSPV', 'LSPV', 'RIPV', 'LIPV', 'MA', 'LAA',
            'Posterior', 'Roof', 'Inferior', 'Lateral',
            'Septal', 'Anterior', 'RSPV_Ostium', 'LSPV_Ostium',
            'RIPV_Ostium', 'LIPV_Ostium'
        ]
        
        def total_discards(region_id):
            return sum(discard_reasons[region_id].values())
        
        results_data = []
        
        print("\n" + "="*100)
        print("WALL THICKNESS RESULTS")
        print("="*100)
        print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'NoHit':<10} {'TooClose':<10} {'TooFar':<10} {'Outlier':<10}")
        print("-"*100)
        
        for region_id in [7, 8, 9, 10, 11, 12]:
            if region_id in thickness_by_region and len(thickness_by_region[region_id]) > 0:
                thicknesses = thickness_by_region[region_id]
                name = region_names[region_id]
                dr = discard_reasons[region_id]
                
                avg_thickness = np.mean(thicknesses)
                std_thickness = np.std(thicknesses)
                num_valid = len(thicknesses)
                
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
        
        print("-"*100)
        print("\nOSTIUM & PV REGIONS")
        print("-"*100)
        
        for region_id in [1, 2, 3, 4, 5, 6, 13, 14, 15, 16]:
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
                        'Category': 'PV/Other',
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
                        'Category': 'PV/Other',
                        'Avg_Thickness_mm': 'N/A',
                        'Std_Dev_mm': 'N/A',
                        'Valid_Vertices': 0,
                        'Discard_NoIntersection': dr['no_intersection'],
                        'Discard_TooClose': dr['too_close'],
                        'Discard_TooFar': dr['too_far'],
                        'Discard_Outlier': dr['outlier'],
                        'Total_Vertices': total_discards(region_id)
                    })
        
        print("-"*100)
        
        return results_data
    
    def add_thickness_to_mesh(self, thickness_per_point):
        """Add thickness values as point data to endocardium mesh."""
        if thickness_per_point is not None and np.any(thickness_per_point >= 0):
            arr = numpy_to_vtk(thickness_per_point)
            arr.SetName("RayThickness")
            pd = self.endo_poly.GetPointData()
            pd.AddArray(arr)
    
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
