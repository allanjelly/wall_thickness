import vtk
import numpy as np
import time
import csv
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


class SimpleWallThickness:
    """Simple KDTree-based wall thickness plugin."""

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
        normal_dot_threshold=0.0,
    ):
        self.endo_poly = endo_poly
        self.epi_poly = epi_poly
        self.grid_bounds = grid_bounds
        self.resolution = resolution
        self.max_thickness_mm = max_thickness_mm
        self.min_thickness_mm = min_thickness_mm
        self.outlier_std_threshold = outlier_std_threshold
        self.normal_dot_threshold = normal_dot_threshold

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
        Calculate wall thickness using KDTree approach and return region stats.
        """
        from scipy.spatial import KDTree
        
        print("\n" + "=" * 60)
        print("  CALCULATING WALL THICKNESS (Simple KDTree Approach)")
        print("=" * 60 + "\n")
        
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
            print("    This may indicate meshes are not aligned.")
            print("    Expected offset for concentric surfaces: < 5mm")
        else:
            print("  ✓ Mesh alignment appears OK (offset < 5mm)")
        
        print("\nFiltering parameters:")
        print(f"  Min thickness: {self.min_thickness_mm} mm (filters mesh overlap/touching)")
        print(f"  Max thickness: {self.max_thickness_mm} mm (filters cross-chamber measurements)")
        print(f"  Normal dot threshold: {self.normal_dot_threshold} (0=accept <90°, 0.5=accept <60°)")
        print(f"  Outlier threshold: {self.outlier_std_threshold} standard deviations")
        
        print("\nRegion categories:")
        print("  Wall regions (7-12): Primary targets for thickness measurement")
        print("  PV regions (1-4): Tubular structures - may have unreliable measurements")
        print("  Ostium regions (13-16): Transition zones")
        print("  Other (5-6): MA, LAA - special structures")
        
        # Build KDTree on exterior surface points
        print("\nBuilding spatial index (KDTree)...")
        t1 = time.time()
        kdtree = KDTree(exterior_points)
        t2 = time.time()
        print(f"  KDTree build time: {(t2 - t1):.2f}s")
        
        # Compute normals for interior mesh vertices
        print("\nComputing interior vertex normals...")
        t1 = time.time()
        normals = self._compute_point_normals()
        t2 = time.time()
        print(f"  Normal computation time: {(t2 - t1):.2f}s")
        
        thickness_by_region = {}
        discard_reasons = {}
        for region_id in range(17):
            thickness_by_region[region_id] = []
            discard_reasons[region_id] = {
                'too_close': 0,
                'too_far': 0,
                'bad_normal': 0,
                'outlier': 0,
            }
        
        # Query nearest points for all interior vertices at once
        print("\nQuerying nearest exterior points...")
        t1 = time.time()
        distances, indices = kdtree.query(interior_points, k=1)
        t2 = time.time()
        query_time = t2 - t1
        print(f"  Query time: {query_time:.2f}s")
        print(f"  Rate: {len(interior_points) / query_time:.0f} vertices/sec")
        
        # Check normal alignment with exterior direction
        print("\nValidating normals against exterior surface...")
        print("  Diagnosing normal direction...")
        t1 = time.time()
        
        dot_products = []
        sample_count = 0
        for i in range(len(interior_points)):
            region_id = interior_regions[i]
            if region_id <= 0:
                continue
            
            nearest_exterior_point = exterior_points[indices[i]]
            to_exterior = nearest_exterior_point - interior_points[i]
            to_exterior_norm = np.linalg.norm(to_exterior)
            
            if to_exterior_norm < 1e-6:
                continue
            
            to_exterior_normalized = to_exterior / to_exterior_norm
            vertex_normal = normals[i]
            dot_product = np.dot(vertex_normal, to_exterior_normalized)
            dot_products.append(dot_product)
            sample_count += 1
            
            if sample_count >= 1000:
                break
        
        if dot_products:
            mean_dot = np.mean(dot_products)
            median_dot = np.median(dot_products)
            print(f"  Dot product statistics (sample of {sample_count} vertices):")
            print(f"    Mean:   {mean_dot:.4f}")
            print(f"    Median: {median_dot:.4f}")
            print(f"    Min:    {np.min(dot_products):.4f}")
            print(f"    Max:    {np.max(dot_products):.4f}")
            
            if median_dot < 0:
                print(f"  ⚠ Normals appear to point INWARD (median dot product: {median_dot:.4f})")
                print("  → Will use abs(dot_product) for validation")
                use_abs_dot = True
            else:
                print(f"  ✓ Normals appear to point OUTWARD (median dot product: {median_dot:.4f})")
                use_abs_dot = False
        else:
            use_abs_dot = False
        
        print("\n  Processing all vertices...")
        thickness_per_point = np.zeros(len(interior_points), dtype=np.float32)
        
        for i in range(len(interior_points)):
            region_id = interior_regions[i]
            if region_id <= 0:
                continue
            
            distance = distances[i]
            
            if distance < self.min_thickness_mm:
                discard_reasons[region_id]['too_close'] += 1
                continue
            
            if distance > self.max_thickness_mm:
                discard_reasons[region_id]['too_far'] += 1
                continue
            
            nearest_exterior_point = exterior_points[indices[i]]
            to_exterior = nearest_exterior_point - interior_points[i]
            to_exterior_norm = np.linalg.norm(to_exterior)
            
            if to_exterior_norm < 1e-6:
                discard_reasons[region_id]['too_close'] += 1
                continue
            
            to_exterior_normalized = to_exterior / to_exterior_norm
            vertex_normal = normals[i]
            dot_product = np.dot(vertex_normal, to_exterior_normalized)
            
            accepted = False
            if use_abs_dot:
                if abs(dot_product) >= self.normal_dot_threshold:
                    accepted = True
                else:
                    discard_reasons[region_id]['bad_normal'] += 1
            else:
                if dot_product >= self.normal_dot_threshold:
                    accepted = True
                else:
                    discard_reasons[region_id]['bad_normal'] += 1
            
            if accepted:
                thickness_by_region[region_id].append(distance)
                thickness_per_point[i] = distance
        
        t2 = time.time()
        print(f"  Processing time: {(t2 - t1):.2f}s")
        
        print("\n  Discard statistics by reason:")
        total_too_close = sum(d['too_close'] for d in discard_reasons.values())
        total_too_far = sum(d['too_far'] for d in discard_reasons.values())
        total_bad_normal = sum(d['bad_normal'] for d in discard_reasons.values())
        print(f"    Too close (<{self.min_thickness_mm}mm): {total_too_close}")
        print(f"    Too far (>{self.max_thickness_mm}mm): {total_too_far}")
        print(f"    Bad normal alignment: {total_bad_normal}")
        
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
            print("  No statistical outliers detected")
        
        def total_discards(region_id):
            return sum(discard_reasons[region_id].values())
        
        region_names = [
            'Background', 'RSPV', 'LSPV', 'RIPV', 'LIPV', 'MA', 'LAA',
            'Posterior', 'Roof', 'Inferior', 'Lateral',
            'Septal', 'Anterior', 'RSPV_Ostium', 'LSPV_Ostium',
            'RIPV_Ostium', 'LIPV_Ostium'
        ]
        
        results_data = []
        
        print("\n" + "=" * 120)
        print("  WALL THICKNESS RESULTS - PRIMARY WALL REGIONS")
        print("=" * 120)
        print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'TooClose':<10} {'TooFar':<10} {'BadNorm':<10} {'Outlier':<10}")
        print("-" * 120)
        
        for region_id in [7, 8, 9, 10, 11, 12]:
            if region_id in thickness_by_region and len(thickness_by_region[region_id]) > 0:
                thicknesses = thickness_by_region[region_id]
                avg_thickness = np.mean(thicknesses)
                std_thickness = np.std(thicknesses)
                num_valid = len(thicknesses)
                name = region_names[region_id]
                dr = discard_reasons[region_id]
                
                print(f"{region_id:<8} {name:<18} {avg_thickness:<12.3f} {std_thickness:<12.3f} {num_valid:<10} {dr['too_close']:<10} {dr['too_far']:<10} {dr['bad_normal']:<10} {dr['outlier']:<10}")
                
                results_data.append({
                    'Region_ID': region_id,
                    'Region_Name': name,
                    'Category': 'Wall',
                    'Avg_Thickness_mm': round(avg_thickness, 4),
                    'Std_Dev_mm': round(std_thickness, 4),
                    'Valid_Vertices': num_valid,
                    'Discard_TooClose': dr['too_close'],
                    'Discard_TooFar': dr['too_far'],
                    'Discard_BadNormal': dr['bad_normal'],
                    'Discard_Outlier': dr['outlier'],
                    'Total_Vertices': num_valid + total_discards(region_id)
                })
        
        print("-" * 120)
        
        print("\n  OSTIUM REGIONS (Transition Zones)")
        print("-" * 120)
        print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'TooClose':<10} {'TooFar':<10} {'BadNorm':<10} {'Outlier':<10}")
        print("-" * 120)
        
        for region_id in [13, 14, 15, 16]:
            if region_id in thickness_by_region and len(thickness_by_region[region_id]) > 0:
                thicknesses = thickness_by_region[region_id]
                avg_thickness = np.mean(thicknesses)
                std_thickness = np.std(thicknesses)
                num_valid = len(thicknesses)
                name = region_names[region_id]
                dr = discard_reasons[region_id]
                
                print(f"{region_id:<8} {name:<18} {avg_thickness:<12.3f} {std_thickness:<12.3f} {num_valid:<10} {dr['too_close']:<10} {dr['too_far']:<10} {dr['bad_normal']:<10} {dr['outlier']:<10}")
                
                results_data.append({
                    'Region_ID': region_id,
                    'Region_Name': name,
                    'Category': 'Ostium',
                    'Avg_Thickness_mm': round(avg_thickness, 4),
                    'Std_Dev_mm': round(std_thickness, 4),
                    'Valid_Vertices': num_valid,
                    'Discard_TooClose': dr['too_close'],
                    'Discard_TooFar': dr['too_far'],
                    'Discard_BadNormal': dr['bad_normal'],
                    'Discard_Outlier': dr['outlier'],
                    'Total_Vertices': num_valid + total_discards(region_id)
                })
        
        print("-" * 120)
        
        print("\n  PV & SPECIAL STRUCTURES (⚠ measurements may be unreliable)")
        print("-" * 120)
        print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'TooClose':<10} {'TooFar':<10} {'BadNorm':<10} {'Outlier':<10}")
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
                    print(f"{region_id:<8} {name:<18} {avg_thickness:<12.3f} {std_thickness:<12.3f} {num_valid:<10} {dr['too_close']:<10} {dr['too_far']:<10} {dr['bad_normal']:<10} {dr['outlier']:<10}")
                    
                    results_data.append({
                        'Region_ID': region_id,
                        'Region_Name': name,
                        'Category': 'PV/Special',
                        'Avg_Thickness_mm': round(avg_thickness, 4),
                        'Std_Dev_mm': round(std_thickness, 4),
                        'Valid_Vertices': num_valid,
                        'Discard_TooClose': dr['too_close'],
                        'Discard_TooFar': dr['too_far'],
                        'Discard_BadNormal': dr['bad_normal'],
                        'Discard_Outlier': dr['outlier'],
                        'Total_Vertices': num_valid + total_discards(region_id)
                    })
                elif total_discards(region_id) > 0:
                    print(f"{region_id:<8} {name:<18} {'N/A':<12} {'N/A':<12} {0:<10} {dr['too_close']:<10} {dr['too_far']:<10} {dr['bad_normal']:<10} {dr['outlier']:<10}")
                    
                    results_data.append({
                        'Region_ID': region_id,
                        'Region_Name': name,
                        'Category': 'PV/Special',
                        'Avg_Thickness_mm': 'N/A',
                        'Std_Dev_mm': 'N/A',
                        'Valid_Vertices': 0,
                        'Discard_TooClose': dr['too_close'],
                        'Discard_TooFar': dr['too_far'],
                        'Discard_BadNormal': dr['bad_normal'],
                        'Discard_Outlier': dr['outlier'],
                        'Total_Vertices': total_discards(region_id)
                    })
        
        print("-" * 120)
        
        self.thickness_per_point = thickness_per_point
        self.thickness_by_region = thickness_by_region
        self.discard_reasons = discard_reasons
        return results_data

    def add_thickness_to_mesh(self, thickness_per_point):
        arr = numpy_to_vtk(thickness_per_point)
        arr.SetName("SimpleThickness")
        pd = self.endo_poly.GetPointData()
        pd.AddArray(arr)
        pd.SetActiveScalars("SimpleThickness")

    def save_output(self, output_path):
        print(f"Saving result to {output_path}")
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(self.endo_poly)
        writer.Write()

    def analyze_regions(self, csv_filename, results_data):
        if not results_data:
            print("No results to save.")
            return

        print(f"\nSaving results to CSV: {csv_filename}")
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['Region_ID', 'Region_Name', 'Category', 'Avg_Thickness_mm', 'Std_Dev_mm',
                             'Valid_Vertices', 'Discard_TooClose', 'Discard_TooFar',
                             'Discard_BadNormal', 'Discard_Outlier', 'Total_Vertices']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_data)
            print(f"✓ Results saved to {csv_filename}")
        except Exception as e:
            print(f"✗ Failed to save CSV: {e}")

    def execute(self, vtk_filename, csv_filename):
        results_data = self.calculate_thickness()
        if results_data is None:
            return False

        self.add_thickness_to_mesh(self.thickness_per_point)
        self.save_output(vtk_filename)
        self.analyze_regions(csv_filename, results_data)
        return True
