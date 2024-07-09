from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
import requests
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon,Point
import numpy as np
from pyproj import Proj, transform
import pyvista as pv
import json
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate", response_class=PlainTextResponse)
async def calculate_sunlight_hours(request: Request, address: str = Form(...)):
    if not address:
        return JSONResponse(content={"error": "No address provided"}, status_code=400)
    print('asdf')
    combined_mesh = get_geometry(address)
    
    # Save the mesh to a JSON file
    #mesh_url = "static/combined_mesh.json"
    #vertices = combined_mesh.points.tolist()
    #faces = combined_mesh.faces.tolist()
    
    #mesh_data = {
    #    "vertices": vertices,
    #    "faces": faces
    #}
    
    # Save the mesh to an STL file
    stl_file_path = "static/combined_mesh.stl"
    
    os.makedirs(os.path.dirname(stl_file_path), exist_ok=True)
    
    try:
        combined_mesh.save(stl_file_path)
        print(f"STL file saved successfully to {stl_file_path}")
    except Exception as e:
        print(f"Error saving STL file: {e}")
        return JSONResponse(content={"error": "Error saving STL file"}, status_code=500)
    
    result_page_url = f"/result?stl_file_path=/{stl_file_path}"
    return result_page_url

@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request, stl_file_path: str):
    print('asdf')
    return templates.TemplateResponse("result.html", {"request": request, "stl_file_path": stl_file_path})

def get_building_height(building, default_height = 10):
    if 'height' in building:
        return float(building['height'])
    elif 'building:levels' in building:
        return float(building['building:levels']) * 3  # Assume average height per level
    else:
        return default_height

def latlon_to_enu(lat, lon, proj_enu):
    """Convert latitude and longitude to ENU coordinates centered at (latitude, longitude)."""
    x, y = proj_enu(lon, lat)
    return x, y

def polygon_to_enu(polygon, height, proj_enu):
    """Convert a shapely Polygon to a pyvista PolyData 3D mesh with ENU coordinates."""
    if not isinstance(polygon, Polygon):
        return None
    
    # Create vertices from the polygon exterior
    exterior_coords = np.array(polygon.exterior.coords)
    enu_coords = np.array([latlon_to_enu(lat, lon, proj_enu) for lon, lat in exterior_coords])
    
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


def get_geometry(address):
    # Placeholder for your script that calculates sunlight hours
    # Use a geocoding service like OpenStreetMap's Nominatim or Google Geocoding API
    location = ox.geocoder.geocode(address)
    latitude, longitude = location[0], location[1]
    
    point = gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326")

    # Buffer the point to create a 100-meter radius
    buffer = point.to_crs(epsg=3857).buffer(100).to_crs(epsg=4326)

    # Download the building footprints within the buffered area
    buildings = ox.geometries_from_polygon(buffer[0], tags={'building': True})

    # Filter to keep only polygons (building footprints)
    buildings = buildings[buildings.geometry.type == 'Polygon']

    buildings['height'] = buildings.apply(get_building_height, axis=1)

    proj_enu = Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=latitude, lon_0=longitude)

    # Convert each building footprint to a pyvista PolyData object
    building_meshes = [polygon_to_enu(geom, height, proj_enu) for geom, height in zip(buildings.geometry, buildings.height) if geom.is_valid]

    # Combine all meshes into a single scene
    combined_mesh = pv.PolyData()
    for mesh in building_meshes:
        combined_mesh += mesh
    
    return combined_mesh
    





