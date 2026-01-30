import vtk
import numpy as np
import time
import csv
from vtk.util.numpy_support import vtk_to_numpy

def calculate_wall_thickness_simple(interior_segmenter, exterior_mesh, regions, 
                             max_thickness_mm=10.0, min_thickness_mm=0.1,
                             outlier_std_threshold=3.0, normal_dot_threshold=0.0):
    """
    Calculate average wall thickness for each region using hybrid KDTree approach.
    
    Hybrid approach:
    1. Use KDTree for fast nearest-point queries
    2. Validate measurements using normal alignment
    3. Apply minimum/maximum anatomical thickness filters
    4. Remove statistical outliers per region
    
    Parameters:
    -----------
    interior_segmenter : LASegmenter
        The segmented interior mesh object
    exterior_mesh : vtkPolyData
        The exterior (epicardium) mesh
    regions : np.ndarray
        The region assignments computed during segmentation
    max_thickness_mm : float
        Maximum plausible wall thickness (default 10mm for atrial wall)
    min_thickness_mm : float
        Minimum plausible wall thickness (default 0.1mm - filters mesh overlap)
    outlier_std_threshold : float
        Number of standard deviations for outlier detection (default 3.0)
    normal_dot_threshold : float
        Minimum dot product between normal and to-exterior direction (default 0.0)
        Lower values accept more measurements; 0.0 accepts any angle < 90°
    """
    from scipy.spatial import KDTree
    
    print("\n" + "="*60)
    print("  CALCULATING WALL THICKNESS (Hybrid Approach)")
    print("="*60 + "\n")
    
    interior_mesh = interior_segmenter.mesh
    interior_points = interior_segmenter.points
    interior_regions = regions
    
    # Extract exterior mesh data as numpy arrays
    exterior_points = vtk_to_numpy(exterior_mesh.GetPoints().GetData())
    
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
    
    # Warn if centroids are far apart (suggesting misalignment)
    if centroid_offset > 5.0:
        print(f"  ⚠ WARNING: Mesh centroids differ by {centroid_offset:.2f}mm!")
        print(f"    This may indicate meshes are not aligned.")
        print(f"    Expected offset for concentric surfaces: < 5mm")
    else:
        print(f"  ✓ Mesh alignment appears OK (offset < 5mm)")
    
    print(f"\nFiltering parameters:")
    print(f"  Min thickness: {min_thickness_mm} mm (filters mesh overlap/touching)")
    print(f"  Max thickness: {max_thickness_mm} mm (filters cross-chamber measurements)")
    print(f"  Normal dot threshold: {normal_dot_threshold} (0=accept <90°, 0.5=accept <60°)")
    print(f"  Outlier threshold: {outlier_std_threshold} standard deviations")
    
    # Define which regions are "true walls" vs "anatomical structures"
    wall_regions = {7, 8, 9, 10, 11, 12}  # Posterior, Roof, Inferior, Lateral, Septal, Anterior
    pv_regions = {1, 2, 3, 4}  # RSPV, LSPV, RIPV, LIPV
    ostium_regions = {13, 14, 15, 16}  # PV ostia
    other_regions = {5, 6}  # MA, LAA
    
    print(f"\nRegion categories:")
    print(f"  Wall regions (7-12): Primary targets for thickness measurement")
    print(f"  PV regions (1-4): Tubular structures - may have unreliable measurements")
    print(f"  Ostium regions (13-16): Transition zones")
    print(f"  Other (5-6): MA, LAA - special structures")
    
    # Generate output filename
    base_path = interior_segmenter.vtk_file.replace('.vtk', '').replace('.wrk', '')
    
    # Build KDTree on exterior surface points
    print("\nBuilding spatial index (KDTree)...")
    t1 = time.time()
    kdtree = KDTree(exterior_points)
    t2 = time.time()
    print(f"  KDTree build time: {(t2-t1):.2f}s")
    
    # Compute normals for interior mesh vertices
    print("\nComputing interior vertex normals...")
    t1 = time.time()
    interior_segmenter.compute_all_vertex_normals()
    normals = interior_segmenter.vertex_normals
    t2 = time.time()
    print(f"  Normal computation time: {(t2-t1):.2f}s")
    
    # Dictionary to store thickness measurements and detailed discard tracking per region
    thickness_by_region = {}
    discard_reasons = {}  # Track WHY vertices were discarded
    for region_id in range(17):
        thickness_by_region[region_id] = []
        discard_reasons[region_id] = {
            'too_close': 0,      # distance < min_thickness (mesh overlap)
            'too_far': 0,        # distance > max_thickness (cross-chamber)
            'bad_normal': 0,     # normal doesn't point toward exterior
            'outlier': 0,        # statistical outlier
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
    
    # Sample first 1000 valid vertices to determine normal direction
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
    
    # Analyze dot product distribution
    if dot_products:
        mean_dot = np.mean(dot_products)
        median_dot = np.median(dot_products)
        print(f"  Dot product statistics (sample of {sample_count} vertices):")
        print(f"    Mean:   {mean_dot:.4f}")
        print(f"    Median: {median_dot:.4f}")
        print(f"    Min:    {np.min(dot_products):.4f}")
        print(f"    Max:    {np.max(dot_products):.4f}")
        
        # Determine if normals point outward or inward
        if median_dot < 0:
            print(f"  ⚠ Normals appear to point INWARD (median dot product: {median_dot:.4f})")
            print(f"  → Will use abs(dot_product) for validation")
            use_abs_dot = True
        else:
            print(f"  ✓ Normals appear to point OUTWARD (median dot product: {median_dot:.4f})")
            use_abs_dot = False
    else:
        use_abs_dot = False
    
    # Now process all vertices with filtering
    print(f"\n  Processing all vertices...")
    
    for i in range(len(interior_points)):
        region_id = interior_regions[i]
        if region_id <= 0:
            continue
        
        distance = distances[i]
        
        # === MIN THICKNESS FILTER (mesh overlap/touching) ===
        if distance < min_thickness_mm:
            discard_reasons[region_id]['too_close'] += 1
            continue
        
        # === MAX THICKNESS FILTER (cross-chamber measurements) ===
        if distance > max_thickness_mm:
            discard_reasons[region_id]['too_far'] += 1
            continue
        
        # Vector from interior to nearest exterior point
        nearest_exterior_point = exterior_points[indices[i]]
        to_exterior = nearest_exterior_point - interior_points[i]
        to_exterior_norm = np.linalg.norm(to_exterior)
        
        if to_exterior_norm < 1e-6:
            discard_reasons[region_id]['too_close'] += 1
            continue
        
        to_exterior_normalized = to_exterior / to_exterior_norm
        vertex_normal = normals[i]
        
        # Check if normal aligns with exterior direction
        dot_product = np.dot(vertex_normal, to_exterior_normalized)
        
        if use_abs_dot:
            # Accept if normal has significant component in either direction
            if abs(dot_product) >= normal_dot_threshold:
                thickness_by_region[region_id].append(distance)
            else:
                discard_reasons[region_id]['bad_normal'] += 1
        else:
            # Only accept if normal points toward exterior
            if dot_product >= normal_dot_threshold:
                thickness_by_region[region_id].append(distance)
            else:
                discard_reasons[region_id]['bad_normal'] += 1
    
    t2 = time.time()
    print(f"  Processing time: {(t2-t1):.2f}s")
    
    # Report discard statistics
    print(f"\n  Discard statistics by reason:")
    total_too_close = sum(d['too_close'] for d in discard_reasons.values())
    total_too_far = sum(d['too_far'] for d in discard_reasons.values())
    total_bad_normal = sum(d['bad_normal'] for d in discard_reasons.values())
    print(f"    Too close (<{min_thickness_mm}mm): {total_too_close}")
    print(f"    Too far (>{max_thickness_mm}mm): {total_too_far}")
    print(f"    Bad normal alignment: {total_bad_normal}")
    
    # === OUTLIER DETECTION PER REGION ===
    print(f"\nApplying outlier detection (>{outlier_std_threshold}σ)...")
    outliers_removed = 0
    
    for region_id in range(1, 17):
        if len(thickness_by_region[region_id]) < 10:
            continue  # Not enough data for meaningful statistics
        
        thicknesses = np.array(thickness_by_region[region_id])
        mean_t = np.mean(thicknesses)
        std_t = np.std(thicknesses)
        
        if std_t < 1e-6:
            continue  # All values are essentially the same
        
        # Find outliers
        lower_bound = mean_t - outlier_std_threshold * std_t
        upper_bound = mean_t + outlier_std_threshold * std_t
        
        # Filter to keep only non-outliers
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
    
    # Calculate total discards per region for reporting
    def total_discards(region_id):
        return sum(discard_reasons[region_id].values())
    
    # Print results to console - separate wall regions from other structures
    region_names = [
        'Background', 'RSPV', 'LSPV', 'RIPV', 'LIPV', 'MA', 'LAA',
        'Posterior', 'Roof', 'Inferior', 'Lateral',
        'Septal', 'Anterior', 'RSPV_Ostium', 'LSPV_Ostium',
        'RIPV_Ostium', 'LIPV_Ostium'
    ]
    
    # Save results to CSV
    csv_filename = base_path + '_wall_thickness.csv'
    results_data = []
    
    # Print WALL REGIONS first (primary results)
    print("\n" + "="*120)
    print("  WALL THICKNESS RESULTS - PRIMARY WALL REGIONS")
    print("="*120)
    print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'TooClose':<10} {'TooFar':<10} {'BadNorm':<10} {'Outlier':<10}")
    print("-" * 120)
    
    for region_id in [7, 8, 9, 10, 11, 12]:  # Wall regions
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
    
    # Print OSTIUM REGIONS
    print("\n  OSTIUM REGIONS (Transition Zones)")
    print("-" * 120)
    print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'TooClose':<10} {'TooFar':<10} {'BadNorm':<10} {'Outlier':<10}")
    print("-" * 120)
    
    for region_id in [13, 14, 15, 16]:  # Ostium regions
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
    
    # Print PV and OTHER REGIONS (less reliable measurements)
    print("\n  PV & SPECIAL STRUCTURES (⚠ measurements may be unreliable)")
    print("-" * 120)
    print(f"{'Region':<8} {'Name':<18} {'Avg (mm)':<12} {'Std Dev':<12} {'Valid':<10} {'TooClose':<10} {'TooFar':<10} {'BadNorm':<10} {'Outlier':<10}")
    print("-" * 120)
    
    for region_id in [1, 2, 3, 4, 5, 6]:  # PV and other regions
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
    
    # Write results to CSV
    if results_data:
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

