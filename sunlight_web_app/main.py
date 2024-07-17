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
import matplotlib.colors as mcolors
import matplotlib.colors as clr

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
    
    sunlight_surf['scaler'] = len(sunlight_surf['sunlight']) * [2]
    sunlight_surf_glyph = sunlight_surf.glyph(scale='scaler',geom=pv.Box())
    sunlight_surf.save('static/output.vtk')
    sunlight_surf_glyph.save('static/output_glpy.vtk')
    
    rbs = linear_colorscheme('static/rbs.json')

    p = pv.Plotter(lighting=None,off_screen=True)
    #p.add_mesh(sunlight_surf_glyph,scalars='sunlight',point_size=10,clim=[0,8],cmap=rbs)
    p.add_mesh(sunlight_surf,scalars='sunlight',clim=[0,12],cmap=rbs)
    p.add_mesh(combined_mesh)
    p.export_gltf(gltf_file_path)

    result_page_url = f"/result?gltf_file_path=/{gltf_file_path}"

    return result_page_url

@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request, gltf_file_path: str):
    return templates.TemplateResponse("result.html", {"request": request, "gltf_file_path": gltf_file_path})
def get_sun_hours(buildings,long,lat):
    grid = get_grid()
    #central_points = grid.cell_centers()
    est = pytz.timezone('US/Eastern')
    hours = np.arange(0,23,1)
    minutes = np.array([0,30])
    total_rec = 0
    bright_field = np.zeros(grid.points.shape[0])
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
 
            if radiation.get_radiation_direct(utc_time, alt_deg) > 200:
                total_rec += 1
                for i,point in enumerate(grid.points):
                    points, ind = buildings.ray_trace(point, sun_coord)
                    if len(points) == 0:
                        bright_field[i] += 1
            
    grid['sunlight'] = bright_field
    grid['sunlight'] /= 2
    #grid['sunlight'] = central_points['sunlight'] / 2
    return grid

def get_grid():
    length = 100  # Length of the grid in meters
    width = 100   # Width of the grid in meters
    resolution = 2  # Resolution of the grid in meters

    # Generate grid points
    x = np.arange(-length, length + resolution, resolution)
    y = np.arange(-length, width + resolution, resolution)
    z = np.zeros((len(x), len(y)))  # Flat surface (z=0)

    # Create a mesh grid
    x, y = np.meshgrid(x, y)
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]

    # Create a PyVista StructuredGrid
    """
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (len(x), len(y), 1)
    """
    grid = pv.PolyData(points)
    grid = grid.delaunay_2d()
    return(grid)

def get_building_height(building, default_height = 5):
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
    
def linear_colorscheme(json_path):
    """Convert the json format linearly spaced color scheme to a matplotlib colormap.
    Args:
        json_path (str): Path of the color scheme in json format. Can be a direct output from paraview
    Returns:
        cmap (LinearSegmentedColormap): Linearly spaced colormap object. Can be used in pyvista plotters
    """
    with open(json_path) as f:
        color_dict = json.load(f)[0]
    # read control points (RGB)
    control_points_rgb = []
    points = []
    if len(color_dict['RGBPoints']) % 4 != 0:
        raise ValueError("RGPPoints length is not correct")
    for i in range(int(len(color_dict['RGBPoints'])/4)):
        control_points_rgb.append((color_dict['RGBPoints'][i*4+1],color_dict['RGBPoints'][i*4+2],color_dict['RGBPoints'][i*4+3]))
        points.append(color_dict['RGBPoints'][i*4])
    points_ratio = [(x - points[0]) / (points[-1] - points[0]) for x in points]
    # interpolate color values between control points based on selected color space
    # paraview also has diverging, lab/CIEDE2000, step but here they are not included
    # first need to convert RGBPoints to the selected color space
    # then interpolate between control points
    # then convert the value from the selected color space back to RGB because matplotlib only takes RGB
    N = 255
    def interp_rgb(mapped_color, points_ratio):
        cmap_data = np.ones((N, 3))
        for i in range(len(mapped_color)-1):
            idx_start = int(points_ratio[i] * N)
            idx_end = int(points_ratio[i+1] * N)
            cmap_data[idx_start:idx_end, 0] = np.linspace(mapped_color[i][0],
                                        control_points_rgb[i+1][0], 
                                        idx_end - idx_start)
            cmap_data[idx_start:idx_end, 1] = np.linspace(mapped_color[i][1],
                                        control_points_rgb[i+1][1], 
                                        idx_end - idx_start)
            cmap_data[idx_start:idx_end, 2] = np.linspace(mapped_color[i][2],
                                        control_points_rgb[i+1][2], 
                                        idx_end - idx_start)
        return(cmap_data)
    def interp_hsv(mapped_color,points_ratio):
        cmap_data = np.ones((N+1, 3))
        x = np.zeros(N+1)
        for i in range(len(points_ratio)-1):
            idx_start = int(points_ratio[i] * N)
            idx_end = int(points_ratio[i+1] * N)
            x[idx_start:(idx_end+1)] = np.linspace(points_ratio[i], points_ratio[i+1], idx_end - idx_start+1)
        cmap_data[:, 0] = np.interp(x, np.linspace(0, 1, len(mapped_color)), [p[0] for p in mapped_color])
        cmap_data[:, 1] = np.interp(x, np.linspace(0, 1, len(mapped_color)), [p[1] for p in mapped_color])
        cmap_data[:, 2] = np.interp(x, np.linspace(0, 1, len(mapped_color)), [p[2] for p in mapped_color])
        return(cmap_data)
    if color_dict['ColorSpace'].lower() == 'lab':
        from colorspacious import cspace_convert
        mapped_color = cspace_convert(control_points_rgb, start='sRGB1', end='CIELab')
        rgb_colors = cspace_convert(interp_rgb(mapped_color,points_ratio), start='CIELab', end='sRGB1')
    elif color_dict['ColorSpace'].lower() == 'hsv':
        mapped_color = [mcolors.rgb_to_hsv(rgb) for rgb in control_points_rgb]
        rgb_colors = [mcolors.hsv_to_rgb(map_data) for map_data in interp_hsv(mapped_color,points_ratio)]
    else:
        rgb_colors = interp_rgb(control_points_rgb,points_ratio)
    cmap = clr.LinearSegmentedColormap.from_list(color_dict['Name'], rgb_colors, N)
    if 'NanColor' in color_dict.keys():
        cmap.set_bad(color = color_dict['NanColor'])
    return (cmap)





