import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon,Point

# Define the address
address = "84 woolwich street, Kitchener, ON"

# Geocode the address to get the latitude and longitude
location = ox.geocoder.geocode(address)
latitude, longitude = location[0], location[1]

# Create a point geometry for the location
point = gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326")

# Buffer the point to create a 100-meter radius
buffer = point.to_crs(epsg=3857).buffer(100).to_crs(epsg=4326)

# Download the building footprints within the buffered area
buildings = ox.geometries_from_polygon(buffer[0], tags={'building': True})

# Filter to keep only polygons (building footprints)
buildings = buildings[buildings.geometry.type == 'Polygon']
#########################
#########################
#########################
#########################
import numpy as np

# Add a default height for buildings without height information
default_height = 10  # meters

def get_building_height(building):
    if 'height' in building:
        return float(building['height'])
    elif 'building:levels' in building:
        return float(building['building:levels']) * 3  # Assume average height per level
    else:
        return default_height

buildings['height'] = buildings.apply(get_building_height, axis=1)

#########################
#########################
#########################
#########################

import numpy as np
from pyproj import Proj, transform
import pyvista as pv

# Define the local ENU projection based on the center latitude and longitude
proj_enu = Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=latitude, lon_0=longitude)

def latlon_to_enu(lat, lon):
    """Convert latitude and longitude to ENU coordinates centered at (latitude, longitude)."""
    x, y = proj_enu(lon, lat)
    return x, y

# Convert building footprints to ENU coordinates
def polygon_to_enu(polygon, height):
    """Convert a shapely Polygon to a pyvista PolyData 3D mesh with ENU coordinates."""
    if not isinstance(polygon, Polygon):
        return None
    
    # Create vertices from the polygon exterior
    exterior_coords = np.array(polygon.exterior.coords)
    enu_coords = np.array([latlon_to_enu(lat, lon) for lon, lat in exterior_coords])
    
    base_vertices = np.hstack([enu_coords, np.zeros((enu_coords.shape[0], 1))])
    top_vertices = np.hstack([enu_coords, np.full((enu_coords.shape[0], 1), height)])
    
    # Combine base and top vertices
    vertices = np.vstack((base_vertices, top_vertices))
    
    # Create faces (two triangles per quad)
    faces = []
    n = len(enu_coords)
    for i in range(n - 1):
        # Create faces for the sides
        faces.append([4, i, i + 1, i + n + 1, i + n])
    
    # Create face for the base and top
    faces.append([n] + list(range(n)))
    faces.append([n] + list(range(n, 2*n)))
    
    faces = np.hstack(faces)
    return pv.PolyData(vertices, faces)
# Convert each building footprint to a pyvista PolyData object
building_meshes = [polygon_to_enu(geom, height) for geom, height in zip(buildings.geometry, buildings.height) if geom.is_valid]

# Combine all meshes into a single scene
combined_mesh = pv.PolyData()
for mesh in building_meshes:
    combined_mesh += mesh
