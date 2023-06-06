from OpenGL.GL import *
import math
import numpy as np
import time
import imgui
import lab_utils as lu
import custom as cu
import copy
from random import uniform
import random

# Light Source Paramaters
g_lightYaw = .0
g_lightPitch = 0
g_lightDistance = 0.0
g_lightColour = lu.vec3(1.0, 1.0, 1.0)
g_lightIntensity = 1.0
g_ambientColour = lu.vec3(1.0, 1.0, 1.0)
g_ambientIntensity = 0.003

# Cameras
g_cameraFree = False
g_freeCamera = cu.FreeCamera([32.0, 18.0, -4.5], 76.0, 28.0)
g_orbitCamera = cu.OrbitCamera([0,0,0], 50, -40.0, -40)
g_selected = 8
g_last_lock = 8
g_justSwapped = False

# Scaling Controls
g_baseScale = 10.0
g_scalePlanets = 10.0
g_scaleDistance = 2.5
g_prev_ScaleDistance = 5.0
g_timeScale = 0.0
g_previous_time = time.time()

# Asteriod Belt Position
g_asteriod_start = 329
g_asteriod_end = 478
g_asteriod_radius = 75
g_asteriod_middle = 403

# Time Controls
g_timeInSeconds = True  

# Used for updating camera
def update(dt, keys, mouseDelta):
    if g_cameraFree:
        g_freeCamera.update(dt, keys, mouseDelta)
    else:
        g_orbitCamera.update(dt, keys, mouseDelta)


def setCameraLock():
    global g_selected
    global g_planets_data
    global g_orbitCamera
    global g_last_lock
    if g_selected != 8:
        selected_planet = g_planets_data[g_selected]
        g_orbitCamera.target = selected_planet["current_position"]
        if g_last_lock != g_selected:
            g_orbitCamera.distance = selected_planet["current_radius"] * 6 
    else:
        g_orbitCamera.target = lu.vec3(0,0,0)
        if g_last_lock != g_selected:
            g_orbitCamera.distance = 50

    if g_last_lock != g_selected:
        g_orbitCamera.yawDeg = -35.0
        g_orbitCamera.pitchDeg = -35.0     
            
    cameraPosition = lu.vec3(0.0, 0.0, g_orbitCamera.distance)
    cameraRotation = lu.Mat3(lu.make_rotation_y(math.radians(g_orbitCamera.yawDeg))) * lu.Mat3(lu.make_rotation_x(math.radians(g_orbitCamera.pitchDeg)))
    rotatedPosition = cameraRotation * cameraPosition
    g_orbitCamera.position = rotatedPosition + g_orbitCamera.target
    g_last_lock = g_selected

# This function is called by the 'magic' to draw a frame width, height are the size of the frame buffer, or window
def renderFrame(width, height):
    global g_previous_time
    global g_planets_data
    global g_justSwapped
    global g_prev_ScaleDistance
    
    if g_timeInSeconds:
        timeFactor = 1.0
    else:
        timeFactor = 60
    
    glViewport(0, 0, width, height)
    # Set the colour we want the frame buffer cleared to, 
    glClearColor(1.0, 1.0, 1.0, 1.0)
    # Tell OpenGL to clear the render target to the clear values for both depth and colour buffers (depth uses the default)
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

    # Time setup, to enable smooth time scaling
    current_time = time.time()
    dt = current_time - g_previous_time
    g_previous_time = current_time

    # Camera setup
    if g_cameraFree:
        if g_justSwapped == True:
            g_freeCamera.position = list(g_orbitCamera.position)
            g_freeCamera.yawDeg = g_orbitCamera.yawDeg
            g_freeCamera.pitchDeg = g_orbitCamera.pitchDeg
            g_justSwapped = False
            
        cameraDirection = lu.Mat3(lu.make_rotation_y(math.radians(g_freeCamera.yawDeg))) * lu.Mat3(lu.make_rotation_x(math.radians(g_freeCamera.pitchDeg))) * [0,0,-1]
        worldToView = lu.make_lookFrom(g_freeCamera.position, cameraDirection, [0,1,0])
    else:
        setCameraLock()
        worldToView = g_orbitCamera.getWorldToViewMatrix([0,1,0])
    
    # Calculate viewToClip transform
    viewToClip = lu.make_perspective(70.0, width / height, 0.1, 12000.0)
    # Calculate base transforms as backup
    modelToView = lu.make_translation(0.0, 0.0, 0.0)
    modelToViewNormal = lu.inverse(lu.transpose(lu.Mat3(modelToView)));

    # Render Skybox
    view_matrix_skybox = copy.deepcopy(worldToView)
    view_matrix_skybox.matData[0, 3] = 0
    view_matrix_skybox.matData[1, 3] = 0
    view_matrix_skybox.matData[2, 3] = 0
    g_skybox.draw(viewToClip, view_matrix_skybox)

    # Light Parameters
    lightRotation = lu.Mat3(lu.make_rotation_y(math.radians(g_lightYaw))) * lu.Mat3(lu.make_rotation_x(math.radians(g_lightPitch))) 
    lightPosition =  lightRotation * lu.vec3(0,0,g_lightDistance)
    lightPosition_inViewSpace = lu.transformPoint(worldToView, lightPosition)  # Transform light position into view space
    lightParameters = [lightPosition_inViewSpace, g_lightColour, g_lightIntensity, g_ambientColour, g_ambientIntensity]


    # Pole correction transform (textures are flipped on the poles)
    correctPoles = lu.make_rotation_x(math.radians(-270))


    # Draw the sun
    sunLightParameters = [lightPosition, g_lightColour, 0.0, g_ambientColour, 1.0]
    spin = lu.make_rotation_y(6 * time.time() * (g_timeScale / timeFactor / 27))
    sunModelToWorld = lu.make_translation(lightPosition[0], lightPosition[1], lightPosition[2]) * spin * lu.make_uniform_scale(g_baseScale) * correctPoles
    g_sunModel.draw(viewToClip * worldToView * sunModelToWorld, modelToView, modelToViewNormal, sunLightParameters)
    
    for planet in g_planets_data:
        # Update orbit position
        planet["orbit_pos"] += dt * g_timeScale / timeFactor / planet["orbit"] * 360
        planet["orbit_pos"] %= 360.0
  
        # Calculate transformations
        orbit = lu.make_rotation_y(math.radians(planet["orbit_pos"]))
        spin = lu.make_rotation_y(6 * time.time() * (g_timeScale / timeFactor / planet["rotation"]))
        distance  = lu.make_translation(0.0, 0.0, planet["dist"] / g_scaleDistance)
        
        # Calculate ModelToWorld transformation
        modelToWorld = orbit * distance * spin * lu.make_uniform_scale(g_baseScale * planet["radius"] * g_scalePlanets) * correctPoles

        # Calculate ModelToView transformation
        modelToView = worldToView * modelToWorld

        # Calculate ModelToViewNormal transformation
        modelToViewNormal = lu.inverse(lu.transpose(lu.Mat3(modelToView)));

        # Update its position and radius for camera locking
        planet["current_position"] = lu.transformPoint(modelToWorld, [0,0,0])
        planet["current_radius"] = planet["radius"] * g_baseScale * g_scalePlanets
        
        # Draw model and orbit ring
        planet["model"].draw(viewToClip * modelToView, modelToView, modelToViewNormal, lightParameters)
        planet["ring"].draw(viewToClip * worldToView * lu.make_scale(planet["dist"] / g_scaleDistance, 1.0, planet["dist"] / g_scaleDistance))
    
    for i, asteriod in enumerate(g_asteriod_data):
        # Update orbit position
        asteriod["orbit_pos"] += dt * g_timeScale / timeFactor / asteriod["orbit"] * 360
        asteriod["orbit_pos"] %= 360.0
  
        # Calculate transformations
        orbit = lu.make_rotation_y(math.radians(asteriod["orbit_pos"]))
        spin = lu.make_rotation_y(6 * time.time() * (g_timeScale / timeFactor / asteriod["rotation"]))
        distance  = lu.make_translation(0.0, 0.0, asteriod["dist"] / g_scaleDistance)
        height = lu.make_translation(0.0, asteriod["height"] / g_scaleDistance, 0.0)
        
        # Calculate ModelToWorld transformation
        modelToWorld = orbit * height * distance * spin * lu.make_uniform_scale(g_baseScale * asteriod["radius"] * g_scalePlanets) 

        asteriod['model'].draw(viewToClip * worldToView * modelToWorld, [[0.56, 0.56, 0.56],  0.15])
    
    if g_scaleDistance != g_prev_ScaleDistance:
        g_dustTorus.update(g_asteriod_middle / g_scaleDistance, g_asteriod_radius / g_scaleDistance)
    
    spin = lu.make_rotation_y(6 * time.time() * (g_timeScale / timeFactor / 730))
    g_dustTorus.draw(viewToClip * worldToView * spin, [[0.33, 0.33, 0.33], 0.072])
    g_prev_ScaleDistance = g_scaleDistance
    
def initResources():
    global g_sunModel
    global g_skybox
    global g_planets_data
    global g_asteriod_data
    global g_dustTorus
    global g_textRender
    
    num_asteriods = 200
        
    g_skybox = cu.SkyBox()
    g_sunModel = cu.Sphere("data/planets/sun.jpg")  
    g_dustTorus = cu.Dust(radius=g_asteriod_middle / g_scaleDistance, tube_radius=g_asteriod_radius / g_scaleDistance)
    # g_textRender = lu.Text("data/fonts/Aaargh.ttf")
    
    g_planets_data = generate_planets_data()
    g_asteriod_data = generate_asteriod_data(num_asteriods)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
    glEnable(GL_FRAMEBUFFER_SRGB)


def generate_asteriod_data(num_rocks):
    g_asteriod_data = []
    for i in range(num_rocks):
        g_asteriod_data.append({
            "model" : cu.SpaceRock(),
            "orbit_pos": i * (360 / num_rocks), 
            "dist": random.randint(329, 478),
            "height" : random.randint(-g_asteriod_radius, g_asteriod_radius),
            "radius": random.uniform(0.0004, 0.0008), 
            "rotation": random.randint(1, 150), 
            "orbit":  random.randint(100, 300)
        })
    return g_asteriod_data

def drawUi():
    global g_cameraFree
    global g_orbitCamera   
    global g_scalePlanets
    global g_scaleDistance
    global g_timeScale
    global g_timeInSeconds 
    global g_justSwapped
    global g_selected
    global g_planets_data
    
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Sun"]


    if imgui.tree_node("Camera Controls", imgui.TREE_NODE_DEFAULT_OPEN):
        g_justSwapped, g_cameraFree = imgui.checkbox("Use Free Camera", g_cameraFree)
        if not g_cameraFree:
            _,g_orbitCamera.distance = imgui.slider_float("CameraDistance", g_orbitCamera.distance, 1.0, 500.0)
            imgui.text("Camera Lock")
            if imgui.radio_button("Sun", g_selected == 8):
                g_selected = 8
            imgui.columns(4)
            for i in range(4):
                if imgui.radio_button(planets[i], g_selected == i):
                    g_selected = i
                if imgui.radio_button(planets[i+4], g_selected == i+4):
                    g_selected = i + 4
                imgui.next_column()
            imgui.columns(1)
        imgui.tree_pop()
    
    if imgui.tree_node("Planet Controls (1.0 = Real)"):
        _,g_scalePlanets = imgui.slider_int("Planet Size", g_scalePlanets, 1.0, 50.0)
        _,g_scaleDistance = imgui.slider_float("Planet Distance", g_scaleDistance, 1.0, 6.0)
        imgui.tree_pop()
    
    if imgui.tree_node("Time Controls "):
        if imgui.radio_button("Seconds", g_timeInSeconds == True):
            g_timeInSeconds = True
        if imgui.radio_button("Minutes", g_timeInSeconds == False):
            g_timeInSeconds = False
        if g_timeInSeconds: 
            _,g_timeScale = imgui.slider_int("1 second = x Days", g_timeScale, 0, 365)
        else:
            _,g_timeScale = imgui.slider_int("1 minute = x Days", g_timeScale, 0, 365)
        imgui.tree_pop()
        
    # ## EXTRA -------------------------------
    # global g_lightYaw
    # global g_lightPitch
    # global g_lightDistance
    # global g_lightColour
    # global g_lightIntensity
    # global g_ambientColour
    # global g_ambientIntensity

    # if imgui.tree_node("Light"):
    #     _,g_lightYaw = imgui.slider_float("Yaw (Deg)", g_lightYaw, -360.00, 360.0)
    #     _,g_lightPitch = imgui.slider_float("Pitch (Deg)", g_lightPitch, -360.00, 360.0)
    #     _,g_lightDistance = imgui.slider_float("Distance", g_lightDistance, -50, 50.0)
    #     _,g_lightColour = lu.imguiX_color_edit3_list("Sun Light Color",  g_lightColour)
    #     _,g_lightIntensity = imgui.slider_float("Sun Light Intensity", g_lightIntensity, 0.0, 1.0)
    #     _,g_ambientColour = lu.imguiX_color_edit3_list("Ambient Color",  g_ambientColour)
    #     _,g_ambientIntensity = imgui.slider_float("Ambient Intensity", g_ambientIntensity, 0.0, 1.0)
    #     imgui.tree_pop()




def generate_planets_data():
    planets_data = [
        {
            "model": cu.Sphere("data/planets/mercury.jpg"), 
            "ring": cu.OrbitRing(), 
            "orbit_pos": 0, 
            "dist": 58, 
            "radius": 0.0033, 
            "rotation": 58.65, 
            "orbit": 88
        },
        {
            "model": cu.Sphere("data/planets/venus.jpg"), 
            "ring": cu.OrbitRing(), 
            "orbit_pos": 0, 
            "dist": 108, 
            "radius": 0.0102, 
            "rotation": 243.02, 
            "orbit": 225
        },
        {
            "model": cu.Sphere("data/planets/earth.jpg"), 
            "ring": cu.OrbitRing(), 
            "orbit_pos": 0, 
            "dist": 150, 
            "radius": 0.0106, 
            "rotation": 1.0, 
            "orbit": 365
        },
        {
            "model": cu.Sphere("data/planets/mars.jpg"), 
            "ring": cu.OrbitRing(), 
            "orbit_pos": 0, 
            "dist": 228, 
            "radius": 0.0056, 
            "rotation": 1.03, 
            "orbit": 687
        },
        {
            "model": cu.Sphere("data/planets/jupiter.jpg"), 
            "ring": cu.OrbitRing(), 
            "orbit_pos": 0, 
            "dist": 778, 
            "radius": 0.1183, 
            "rotation": 0.415, 
            "orbit": 4333
        },
        {
            "model": cu.Sphere("data/planets/saturn.jpg"), 
            "ring": cu.OrbitRing(), 
            "orbit_pos": 0, 
            "dist": 1429, 
            "radius": 0.1003, 
            "rotation": 0.445, 
            "orbit": 10759
        },
        {
            "model": cu.Sphere("data/planets/uranus.jpg"), 
            "ring": cu.OrbitRing(), 
            "orbit_pos": 0, 
            "dist": 2871, 
            "radius": 0.0428, 
            "rotation": 0.718, 
            "orbit": 30687
        },
        {
            "model": cu.Sphere("data/planets/neptune.jpg"), 
            "ring": cu.OrbitRing(), 
            "orbit_pos": 0, 
            "dist": 4504, 
            "radius": 0.0411, 
            "rotation": 0.671, 
            "orbit": 60190
        }
    ]
    return planets_data


# This does all the openGL setup and window creation needed
# it hides a lot of things that we will want to get a handle on as time goes by.
lu.runProgram("Solar System", 1280, 800, renderFrame, initResources, drawUi, update)


