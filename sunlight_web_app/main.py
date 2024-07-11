from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon,Point
import numpy as np
from pyproj import Proj, transform
import pyvista as pv
import json
import os
import pytz
from pysolar.solar import *
import datetime
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate", response_class=PlainTextResponse)
async def calculate_sunlight_hours(request: Request, address: str = Form(...)):
    if not address:
        return JSONResponse(content={"error": "No address provided"}, status_code=400)
    combined_mesh,long,lat = get_geometry(address)

    sunlight_surf = get_sun_hours(combined_mesh,long,lat)
    
    gltf_file_path = 'static/output.gltf'
    #sunlight_surf.save('static/test.vtk')
    sunlight_surf['scaler'] = len(sunlight_surf['sunlight']) * [2]
    sunlight_surf_glyph = sunlight_surf.glyph(scale='scaler',geom=pv.Box())

    p = pv.Plotter(lighting=None,off_screen=True)
    p.add_mesh(sunlight_surf_glyph,scalars='sunlight',point_size=10,clim=[0,12])
    p.add_mesh(combined_mesh)
    p.export_gltf(gltf_file_path)

    result_page_url = f"/result?gltf_file_path=/{gltf_file_path}"

    return result_page_url

@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request, gltf_file_path: str):
    return templates.TemplateResponse("result.html", {"request": request, "gltf_file_path": gltf_file_path})
def get_sun_hours(buildings,long,lat):
    grid = get_grid()
    central_points = grid.cell_centers()
    est = pytz.timezone('US/Eastern')
    hours = np.arange(0,23,1)
    minutes = np.array([0,30])
    total_rec = 0
    bright_field = np.zeros(central_points.points.shape[0])
    for i_h, hr_ in enumerate(hours):
        for i_m, min_ in enumerate(minutes):
            est_time = est.localize(datetime.datetime(2023, 7, 1, hr_, min_, 0))  # Replace with your desired EST time
            # Convert EST time to UTC
            utc_time = est_time.astimezone(pytz.utc)
            #get sun location
            d_sun = 1.5e11
            alt_deg = get_altitude(lat, long, utc_time)
            az_deg = get_azimuth(lat, long, utc_time)

            sun_coord = np.array([np.cos(np.deg2rad(az_deg)) * d_sun, 
                                np.sin(np.deg2rad(az_deg)) * d_sun, 
                                np.sin(np.deg2rad(alt_deg)) * d_sun])
 
            cell_ind = []
            #print(radiation.get_radiation_direct(utc_time, alt_deg))
            if radiation.get_radiation_direct(utc_time, alt_deg) > 0: 
                total_rec += 1
                for i,point in enumerate(central_points.points):
                    points, ind = buildings.ray_trace(point, sun_coord)
                    if len(points) == 0:
                        bright_field[i] += 1
            
    central_points['sunlight'] = bright_field

    #grid = grid.interpolate(central_points)
    return central_points

def get_grid():
    length = 100  # Length of the grid in meters
    width = 100   # Width of the grid in meters
    resolution = 2.5  # Resolution of the grid in meters

    # Generate grid points
    x = np.arange(-length, length + resolution, resolution)
    y = np.arange(-length, width + resolution, resolution)
    z = np.zeros((len(x), len(y)))  # Flat surface (z=0)

    # Create a mesh grid
    x, y = np.meshgrid(x, y)
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]

    # Create a PyVista StructuredGrid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (len(x), len(y), 1)
    return(grid)

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
    
    return (combined_mesh, longitude,latitude)
    





