"""Distance and Progression Calculation Services"""
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional


class DistanceCalculatorService:
    """
    Service for calculating distances between fovea and GA regions.
    Uses pixel-to-micron conversion based on the 1800 micron optic disc standard.
    """
    
    def __init__(self):
        """Initialize distance calculator service."""
        pass
    
    def calculate_fovea_to_ga_distance(
        self,
        fovea_x: float,
        fovea_y: float,
        ga_region: List[Tuple[int, int]],
        pixel_to_micron_ratio: float
    ) -> Dict[str, float]:
        """
        Calculate the shortest distance from fovea to a GA region.
        
        Args:
            fovea_x: X coordinate of fovea
            fovea_y: Y coordinate of fovea
            ga_region: List of (x, y) points defining the GA region contour
            pixel_to_micron_ratio: Conversion factor (1800 / disc_height_pixels)
        
        Returns:
            Dictionary containing:
                - distance_pixels: Distance in pixels
                - distance_microns: Distance in microns
                - nearest_ga_point_x: X coordinate of nearest GA point
                - nearest_ga_point_y: Y coordinate of nearest GA point
        """
        fovea_pt = np.array([fovea_x, fovea_y])
        ga_points = np.array(ga_region)
        
        # Calculate Euclidean distance to all points on GA boundary
        distances = np.sqrt(np.sum((ga_points - fovea_pt)**2, axis=1))
        
        # Find minimum distance
        min_idx = np.argmin(distances)
        min_distance_pixels = distances[min_idx]
        nearest_point = ga_points[min_idx]
        
        # Convert to microns
        min_distance_microns = min_distance_pixels * pixel_to_micron_ratio
        
        print(f"  [DistanceCalculator] Fovea to GA: {min_distance_pixels:.1f} px = {min_distance_microns:.1f} µm")
        
        return {
            'distance_pixels': float(min_distance_pixels),
            'distance_microns': float(min_distance_microns),
            'nearest_ga_point_x': int(nearest_point[0]),
            'nearest_ga_point_y': int(nearest_point[1])
        }
    
    def calculate_distances_to_all_regions(
        self,
        fovea_x: float,
        fovea_y: float,
        ga_regions: List[List[Tuple[int, int]]],
        pixel_to_micron_ratio: float
    ) -> List[Dict[str, float]]:
        """
        Calculate distances from fovea to all GA regions.
        
        Args:
            fovea_x: X coordinate of fovea
            fovea_y: Y coordinate of fovea
            ga_regions: List of GA regions, each as list of (x, y) points
            pixel_to_micron_ratio: Conversion factor
        
        Returns:
            List of distance dictionaries, one per region
        """
        results = []
        for i, region in enumerate(ga_regions):
            distance_info = self.calculate_fovea_to_ga_distance(
                fovea_x, fovea_y, region, pixel_to_micron_ratio
            )
            distance_info['region_index'] = i
            results.append(distance_info)
        
        # Sort by distance (closest first)
        results.sort(key=lambda x: x['distance_microns'])
        
        return results


class ProgressionCalculatorService:
    """
    Service for calculating GA progression rate and predicting foveal involvement.
    Implements the progression logic from README.md (lines 118-133).
    """
    
    def __init__(self):
        """Initialize progression calculator service."""
        pass
    
    def calculate_progression(
        self,
        date_before: str,
        date_after: str,
        distance_before_microns: float,
        distance_after_microns: float,
        eye_side_before: str,
        eye_side_after: str
    ) -> Dict:
        """
        Calculate GA progression rate and predict foveal involvement date.
        
        Args:
            date_before: ISO date string (YYYY-MM-DD) for before image
            date_after: ISO date string (YYYY-MM-DD) for after image
            distance_before_microns: Fovea-to-GA distance in before image (microns)
            distance_after_microns: Fovea-to-GA distance in after image (microns)
            eye_side_before: "OD" or "OS" for before image
            eye_side_after: "OD" or "OS" for after image
        
        Returns:
            Dictionary containing progression analysis results
        
        Raises:
            ValueError: If dates are invalid or eyes don't match
        """
        # Validation: Check same eye
        if eye_side_before != eye_side_after:
            return {
                'status': 'error',
                'error_message': f"Eye mismatch: Before is {eye_side_before}, After is {eye_side_after}",
                'days_elapsed': 0,
                'distance_change_microns': 0.0,
                'rate_microns_per_day': None,
                'rate_microns_per_month': None,
                'predicted_foveal_involvement_date': None
            }
        
        # Parse dates
        try:
            dt_before = datetime.fromisoformat(date_before)
            dt_after = datetime.fromisoformat(date_after)
        except ValueError as e:
            return {
                'status': 'error',
                'error_message': f"Invalid date format: {str(e)}",
                'days_elapsed': 0,
                'distance_change_microns': 0.0,
                'rate_microns_per_day': None,
                'rate_microns_per_month': None,
                'predicted_foveal_involvement_date': None
            }
        
        # Validation: After date must be after before date
        if dt_after <= dt_before:
            return {
                'status': 'error',
                'error_message': "After date must be later than before date",
                'days_elapsed': 0,
                'distance_change_microns': 0.0,
                'rate_microns_per_day': None,
                'rate_microns_per_month': None,
                'predicted_foveal_involvement_date': None
            }
        
        # Calculate time elapsed
        time_elapsed = dt_after - dt_before
        days_elapsed = time_elapsed.days
        
        # Calculate distance change (POSITIVE = progression toward fovea)
        distance_change = distance_before_microns - distance_after_microns
        
        print(f"  [ProgressionCalculator] Time elapsed: {days_elapsed} days")
        print(f"  [ProgressionCalculator] Distance change: {distance_change:.1f} µm")
        
        # Three cases per README.md logic
        
        if distance_change > 0:
            # CASE 1: Progression detected
            rate_per_day = distance_change / days_elapsed
            rate_per_month = rate_per_day * 30.44  # Average days per month
            
            # Predict foveal involvement date
            days_to_fovea = distance_after_microns / rate_per_day
            predicted_date = dt_after + timedelta(days=days_to_fovea)
            predicted_date_str = predicted_date.strftime('%Y-%m-%d')
            
            print(f"  [ProgressionCalculator] Rate: {rate_per_day:.3f} µm/day ({rate_per_month:.1f} µm/month)")
            print(f"  [ProgressionCalculator] Predicted foveal involvement: {predicted_date_str}")
            
            return {
                'status': 'progression',
                'error_message': None,
                'days_elapsed': days_elapsed,
                'distance_change_microns': round(distance_change, 2),
                'rate_microns_per_day': round(rate_per_day, 3),
                'rate_microns_per_month': round(rate_per_month, 2),
                'predicted_foveal_involvement_date': predicted_date_str
            }
        
        elif distance_change == 0:
            # CASE 2: No progression
            print("  [ProgressionCalculator] No progression detected")
            
            return {
                'status': 'no_progression',
                'error_message': None,
                'days_elapsed': days_elapsed,
                'distance_change_microns': 0.0,
                'rate_microns_per_day': 0.0,
                'rate_microns_per_month': 0.0,
                'predicted_foveal_involvement_date': None
            }
        
        else:
            # CASE 3: Negative progression (ERROR)
            # GA appears further from fovea - likely measurement error
            print(f"  [ProgressionCalculator] ERROR: Negative progression ({distance_change:.1f} µm)")
            
            return {
                'status': 'error',
                'error_message': f"Negative progression detected ({distance_change:.1f} µm). GA appears further from fovea. Check measurements.",
                'days_elapsed': days_elapsed,
                'distance_change_microns': round(distance_change, 2),
                'rate_microns_per_day': None,
                'rate_microns_per_month': None,
                'predicted_foveal_involvement_date': None
            }
