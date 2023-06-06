from OpenGL.GL import *
import numpy as np
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at
import xml.etree.ElementTree as ET
import math
import imgui
import random
import freetype
from PIL import Image
from collections import deque
from noise import snoise3
from lab_utils import *

def bindAndSetTexture(texUnit, textureId, textureName, shaderProgram,  textureType = GL_TEXTURE_2D):
    glUseProgram(shaderProgram)
    bindTexture(texUnit, textureId, textureType)
    loc = glGetUniformLocation(shaderProgram, textureName)
    glUniform1i(loc, texUnit)
    glUseProgram(0)

def bindTexture(texUnit, textureId, textureType = GL_TEXTURE_2D):
	glActiveTexture(GL_TEXTURE0 + texUnit);
	glBindTexture(textureType, textureId);

def loadtexture(fileName): 
    with Image.open(fileName) as image:
        texId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texId)

        data = image.tobytes("raw", "RGBX" if image.mode == 'RGB' else "RGBA", 0, -1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.size[0], image.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

        glGenerateMipmap(GL_TEXTURE_2D)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        glBindTexture(GL_TEXTURE_2D, 0);
        return texId


class FreeCamera:
    position = vec3(0.0,0.0,0.0)
    yawDeg = 0.0
    pitchDeg = 0.0
    maxSpeed = 50
    angSpeed = 10

    def __init__(self, pos, yawDeg, pitchDeg):
        self.position = vec3(*pos)
        self.yawDeg = yawDeg
        self.pitchDeg = pitchDeg
        self.mouseDeltaBuffer = deque(maxlen=5)

    def update(self, dt, keys, mouseDelta):
        cameraSpeed = 0.0
        cameraTurnSpeed = 0.0
        cameraPitchSpeed = 0.0
        cameraStrafeSpeed = 0.0

        if keys["UP"] or keys["W"]:
            cameraSpeed -= self.maxSpeed
        if keys["DOWN"] or keys["S"]:
            cameraSpeed += self.maxSpeed
        if keys["LEFT"]:
            cameraTurnSpeed += self.angSpeed
        if keys["RIGHT"]:
            cameraTurnSpeed -= self.angSpeed
        if keys["A"]:
            cameraStrafeSpeed -= self.maxSpeed
        if keys["D"]:
            cameraStrafeSpeed += self.maxSpeed

        # Mouse look is enabled with right mouse button
        if keys["MOUSE_BUTTON_LEFT"]:
            self.mouseDeltaBuffer.append(mouseDelta)
            mouseDeltaAvg = np.mean(self.mouseDeltaBuffer, axis=0)
            cameraTurnSpeed = mouseDeltaAvg[0] * self.angSpeed
            cameraPitchSpeed = mouseDeltaAvg[1] * self.angSpeed

        self.yawDeg += cameraTurnSpeed * dt
        self.pitchDeg = min(89.0, max(-89.0, self.pitchDeg - cameraPitchSpeed * dt))

        cameraRotation = Mat3(make_rotation_y(math.radians(self.yawDeg))) * Mat3(make_rotation_x(math.radians(self.pitchDeg))) 
        cameraDirection = cameraRotation * [0,0,1]
        self.position += np.array(cameraDirection) * cameraSpeed * dt

        #strafe measns perpendicular left-right movement, so rotate the X unit vector and go
    
        self.position += np.array(cameraRotation * [1,0,0]) * cameraStrafeSpeed * dt

    def drawUi(self):
        if imgui.tree_node("FreeCamera", imgui.TREE_NODE_DEFAULT_OPEN):
            _,self.yawDeg = imgui.slider_float("Yaw (Deg)", self.yawDeg, -180.00, 180.0)
            _,self.pitchDeg = imgui.slider_float("Pitch (Deg)", self.pitchDeg, -89.00, 89.0)
            imgui.tree_pop()
    
    def getWorldToViewMatrix(self, up):
        cameraDirection = Mat3(make_rotation_y(math.radians(self.yawDeg))) * Mat3(make_rotation_x(math.radians(self.pitchDeg))) * [0,0,-1]
        return make_lookFrom(self.position, cameraDirection, [0,1,0])
    

class OrbitCamera:
    target = vec3(0.0,0.0,0.0)
    distance = 1.0
    yawDeg = 0.0
    pitchDeg = 0.0
    maxSpeed = 10
    angSpeed = 10
    position = vec3(0.0,1.0,0.0)

    def __init__(self, target, distance, yawDeg, pitchDeg):
        self.target = vec3(*target)
        self.yawDeg = yawDeg
        self.pitchDeg = pitchDeg
        self.distance = distance
        self.mouseDeltaBuffer = deque(maxlen=5)

    def update(self, dt, keys, mouseDelta):
        cameraSpeed = 0.0
        cameraTurnSpeed = 0.0
        cameraPitchSpeed = 0.0
        cameraStrafeSpeed = 0.0

        # Mouse look is enabled with right mouse button
        if keys["MOUSE_BUTTON_LEFT"]:
            self.mouseDeltaBuffer.append(mouseDelta)
            mouseDeltaAvg = np.mean(self.mouseDeltaBuffer, axis=0)
            cameraTurnSpeed = mouseDeltaAvg[0] * self.angSpeed
            cameraPitchSpeed = mouseDeltaAvg[1] * self.angSpeed

        if keys["MOUSE_BUTTON_RIGHT"]:
            self.distance = max(1.0, self.distance + mouseDelta[1])

        self.yawDeg += cameraTurnSpeed * dt
        self.pitchDeg = min(89.0, max(-89.0, self.pitchDeg + cameraPitchSpeed * dt))

        cameraRotation = Mat3(make_rotation_y(math.radians(self.yawDeg))) * Mat3(make_rotation_x(math.radians(self.pitchDeg))) 
        self.position = cameraRotation * vec3(0,0,self.distance)

    def drawUi(self):
        if imgui.tree_node("OrbitCamera", imgui.TREE_NODE_DEFAULT_OPEN):
            _,self.yawDeg = imgui.slider_float("Yaw (Deg)", self.yawDeg, -180.00, 180.0)
            _,self.pitchDeg = imgui.slider_float("Pitch (Deg)", self.pitchDeg, -89.00, 89.0)
            _,self.distance = imgui.slider_float("Distance", self.distance, 1.00, 1000.0)
            imgui.tree_pop()
    
    def getWorldToViewMatrix(self, up):
        return make_lookAt(self.position, self.target, up)

g_texture_count = -1
class Sphere:
    def __init__(self, texture, r=1.0, sectors=36, stacks=18):
        global g_texture_count
        self.radius = r
        self.sectorCount = sectors
        self.stackCount = stacks
        self.sphere_vertices = []
        self.sphere_indices = []
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)
        self.shader = None
        g_texture_count += 1
        self.textUnit = g_texture_count
        self.textId = 0
        
        self.generate_vertices()
        self.generate_indices()
        self.generate_buffers()
        self.generate_shader()
        self.construct_texture(texture)
        
    def generate_vertices(self):
        for i in range(self.stackCount + 1):
            stack_angle = math.pi / 2 - i * math.pi / self.stackCount
            xy = self.radius * math.cos(stack_angle)
            z = self.radius * math.sin(stack_angle)

            for j in range(self.sectorCount + 1):
                sector_angle = j * 2 * math.pi / self.sectorCount

                x = xy * math.cos(sector_angle)
                y = xy * math.sin(sector_angle)
                nx, ny, nz = x / self.radius, y / self.radius, z / self.radius
                s = j / self.sectorCount
                t = i / self.stackCount
                self.sphere_vertices.extend([x, y, z, nx, ny, nz, s, t])


    def generate_indices(self):
        for i in range(self.stackCount):
            k1 = i * (self.sectorCount + 1)
            k2 = k1 + self.sectorCount + 1

            for j in range(self.sectorCount):
                if i != 0:
                    self.sphere_indices.extend([k1, k2, k1 + 1])

                if i != (self.stackCount-1):
                    self.sphere_indices.extend([k1 + 1, k2, k2 + 1])

                k1 += 1
                k2 += 1

    def generate_buffers(self):
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 
                     np.array(self.sphere_vertices, dtype=np.float32), 
                     GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
                     np.array(self.sphere_indices, dtype=np.uint32), 
                     GL_DYNAMIC_DRAW)

        # Set the position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, None)
        glEnableVertexAttribArray(0)

        # Set the normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        # Set the texture attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def construct_texture(self, texture):
        self.textId = loadtexture(texture)
        bindAndSetTexture(self.textUnit, self.textId, "texture1", self.shader)

    def draw(self, modelToClipTransform, modelToViewTransform, modelToViewNormalTransform, lightParameters):
        glUseProgram(self.shader)
        setUniform(self.shader, "modelToClipTransform", modelToClipTransform)
        setUniform(self.shader, "modelToViewTransform", modelToViewTransform)
        setUniform(self.shader, "modelToViewNormalTransform", modelToViewNormalTransform)  
        setUniform(self.shader, "lightPosition", lightParameters[0])
        setUniform(self.shader, "lightColor", lightParameters[1])
        setUniform(self.shader, "lightIntensity", lightParameters[2])
        setUniform(self.shader, "ambientColor", lightParameters[3])
        setUniform(self.shader, "ambientIntensity", lightParameters[4])
                   
        glActiveTexture(GL_TEXTURE0 + self.textUnit)
        glBindTexture(GL_TEXTURE_2D, self.textId)
        
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.sphere_indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glUseProgram(0)
        
    def generate_shader(self):
        vertexShader = """
            #version 330
            in vec3 positionIn;
            in vec3 normalIn;
            in vec2	texCoord;

            uniform mat4 modelToClipTransform;
            uniform mat4 modelToViewTransform;
            uniform mat3 modelToViewNormalTransform;
            
            out VertexData
            {
                vec3 v2f_viewSpaceNormal;
                vec3 v2f_viewSpacePosition;
                vec2 v2f_texCoord;
            };


            void main() 
            {
                v2f_viewSpacePosition = (modelToViewTransform * vec4(positionIn, 1.0)).xyz;
                v2f_viewSpaceNormal = normalize(modelToViewNormalTransform * normalIn);
                v2f_texCoord = texCoord;
	            gl_Position = modelToClipTransform * vec4(positionIn, 1.0);
            }"""
            
        fragmentShader = """
            #version 330
            in VertexData
            {
                vec3 v2f_viewSpaceNormal;
                vec3 v2f_viewSpacePosition;
                vec2 v2f_texCoord;
            };

            uniform sampler2D texture1;
            uniform vec3 lightPosition;
            uniform vec3 lightColor;
            uniform float lightIntensity;
            uniform vec3 ambientColor;
            uniform float ambientIntensity;

            out vec4 fragmentColor;

            void main() 
            {
                vec3 toLight = lightPosition - v2f_viewSpacePosition;
                float distanceToLight = length(toLight);
                vec3 lightDirection = toLight / distanceToLight;
                float diffuseFactor = max(dot(v2f_viewSpaceNormal, lightDirection), 0.0);
                vec3 diffuseColor = diffuseFactor * lightColor * lightIntensity / (distanceToLight * 0.05);
                vec3 ambientColorTotal = ambientColor * ambientIntensity;
                vec4 textureColor = texture(texture1, v2f_texCoord);
                fragmentColor = vec4((diffuseColor + ambientColorTotal), 1.0) * textureColor;
            }
        """
        self.shader = buildShader([vertexShader], [fragmentShader], {"positionIn" : 0, "normalIn" : 1, "texCoord" : 2})

class SpaceRock():
    def __init__(self, r=1.0, sectors=3, stacks=3, noise_scale=0.2, noise_amplitude=0.2):
        self.noise_scale = noise_scale
        self.noise_amplitude = noise_amplitude
        self.radius = r
        self.sectorCount = sectors
        self.stackCount = stacks
        self.sphere_vertices = []
        self.sphere_indices = []
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)
        
        self.generate_vertices()
        self.generate_indices()
        self.generate_buffers()
        self.generate_shader()


    def generate_vertices(self):
        for i in range(self.stackCount + 1):
            stack_angle = math.pi / 2 - i * math.pi / self.stackCount
            xy = self.radius * math.cos(stack_angle)
            z = self.radius * math.sin(stack_angle)

            for j in range(self.sectorCount + 1):
                sector_angle = j * 2 * math.pi / self.sectorCount

                x = xy * math.cos(sector_angle)
                y = xy * math.sin(sector_angle)

                # add noise to the vertex positions
                noise = snoise3(x * self.noise_scale, y * self.noise_scale, z * self.noise_scale)
                x += noise * self.noise_amplitude
                y += noise * self.noise_amplitude
                z += noise * self.noise_amplitude

                self.sphere_vertices.extend([x, y, z])

    def generate_indices(self):
        for i in range(self.stackCount):
            k1 = i * (self.sectorCount + 1)
            k2 = k1 + self.sectorCount + 1

            for j in range(self.sectorCount):
                if i != 0:
                    self.sphere_indices.extend([k1, k2, k1 + 1])

                if i != (self.stackCount-1):
                    self.sphere_indices.extend([k1 + 1, k2, k2 + 1])

                k1 += 1
                k2 += 1

    def generate_buffers(self):
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 
                     np.array(self.sphere_vertices, dtype=np.float32), 
                     GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
                     np.array(self.sphere_indices, dtype=np.uint32), 
                     GL_DYNAMIC_DRAW)

        # Set the position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
        glEnableVertexAttribArray(0)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)


    def draw(self, modelToClipTransform, lightParameters):
        glUseProgram(self.shader)
        setUniform(self.shader, "modelToClipTransform", modelToClipTransform)
        setUniform(self.shader, "ambientColor", lightParameters[0])
        setUniform(self.shader, "ambientIntensity", lightParameters[1])
                           
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.sphere_indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glUseProgram(0)


    def generate_shader(self):
        vertexShader = """
            #version 330
            in vec3 positionIn;

            uniform mat4 modelToClipTransform;            
            void main() 
            {
                gl_Position = modelToClipTransform * vec4(positionIn, 1.0);
            }"""

    
        fragmentShader = """
            #version 330
            uniform vec3 ambientColor;
            uniform float ambientIntensity;

            out vec4 fragmentColor;

            void main() 
            {
                vec3 ambientColorTotal = ambientColor * ambientIntensity;
                fragmentColor = vec4((ambientColorTotal), 1.0);
            }
            """
        self.shader = buildShader([vertexShader], [fragmentShader], {"positionIn" : 0})

class Dust():
    def __init__(self, count=30000, radius=1.0, tube_radius=2.0):
        self.count = count
        self.radius = radius
        self.tube_radius = tube_radius
        self.dust_vertices = []
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        
        self.generate_vertices()
        self.generate_buffers()
        self.generate_shader()


    def generate_vertices(self):
        for _ in range(self.count):
            theta = 2 * math.pi * random.random() 
            phi = 2 * math.pi * random.random()
            
            minor_radius = math.sqrt(random.random()) * self.tube_radius
            major_radius = self.radius + minor_radius * math.cos(phi)
            
            x = major_radius * math.cos(theta)
            z = major_radius * math.sin(theta)
            y = minor_radius * math.sin(phi)
            
            self.dust_vertices.extend([x, y, z])


    def update(self, radius, tube_radius):
            self.radius = radius
            self.tube_radius = tube_radius
            self.dust_vertices = []
            self.generate_vertices()
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferData(GL_ARRAY_BUFFER, 
                        np.array(self.dust_vertices, dtype=np.float32), 
                        GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def generate_buffers(self):
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 
                     np.array(self.dust_vertices, dtype=np.float32), 
                     GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
        glEnableVertexAttribArray(0)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)


    def draw(self, modelToClipTransform, lightParameters):
        glUseProgram(self.shader)
        setUniform(self.shader, "modelToClipTransform", modelToClipTransform)
        setUniform(self.shader, "ambientColor", lightParameters[0])
        setUniform(self.shader, "ambientIntensity", lightParameters[1])
                           
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_POINTS, 0, self.count)
        glBindVertexArray(0)
        glUseProgram(0)


    def generate_shader(self):
        vertexShader = """
            #version 330
            in vec3 positionIn;

            uniform mat4 modelToClipTransform;            
            void main() 
            {
                gl_Position = modelToClipTransform * vec4(positionIn, 1.0);
                gl_PointSize = 1.0;  // Adjust size as needed
            }"""

        fragmentShader = """
            #version 330
            uniform vec3 ambientColor;
            uniform float ambientIntensity;

            out vec4 fragmentColor;

            void main() 
            {
                vec3 ambientColorTotal = ambientColor * ambientIntensity;
                fragmentColor = vec4((ambientColorTotal), 1.0);
            }
            """
        self.shader = buildShader([vertexShader], [fragmentShader], {"positionIn" : 0})


class OrbitRing:
    def __init__(self, r=1.0, sectors=100):
        self.radius = r
        self.sectorCount = sectors
        self.orbit_vertices = []
        self.orbit_indices = []
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)
        self.shader = None

        self.generate_vertices()
        self.generate_indices()
        self.generate_buffers()
        self.generate_shader()

    def generate_vertices(self):
        for i in range(self.sectorCount + 1):
            sector_angle = i * 2 * math.pi / self.sectorCount

            x = self.radius * math.cos(sector_angle)
            y = 0
            z = self.radius * math.sin(sector_angle)
            self.orbit_vertices.extend([x, y, z])

    def generate_indices(self):
        for i in range(self.sectorCount):
            if i != self.sectorCount - 1:
                self.orbit_indices.extend([i, i + 1])
            else:
                self.orbit_indices.extend([i, 0])

    def generate_buffers(self):
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 
                     np.array(self.orbit_vertices, dtype=np.float32), 
                     GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
                     np.array(self.orbit_indices, dtype=np.uint32), 
                     GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw(self, modelToClipTransform, color=[1.0, 1.0, 1.0]):
        glUseProgram(self.shader)
        setUniform(self.shader, "modelToClipTransform", modelToClipTransform)
        setUniform(self.shader, "color", color)
        
        glBindVertexArray(self.VAO)
        glDrawElements(GL_LINES, len(self.orbit_indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glUseProgram(0)
        
    def generate_shader(self):
        vertexShader = """
            #version 330
            in vec3 positionIn;

            uniform mat4 modelToClipTransform;

            void main() 
            {
                gl_Position = modelToClipTransform * vec4(positionIn, 1.0);
            }"""
            
        fragmentShader = """
            #version 330

            uniform vec3 color;

            out vec4 fragmentColor;

            void main() 
            {
                fragmentColor = vec4(color, 1.0);
            }
        """
        self.shader = buildShader([vertexShader], [fragmentShader], {"positionIn" : 0})

class SkyBox: 
    
    def __init__(self, textures=['right.png', 'left.png', 'top.png', 'bottom.png', 'back.png', 'front.png']):
        self.texId = None
        self.vertices = None
        self.shader = None
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        
        self.load_skybox(textures)
        self.generate_vertices()
        self.build_skybox()
        self.generate_shader()
    
    def load_skybox(self, textures):
        self.texId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texId)

        for i in range(6):
            image = Image.open("data/skybox/" + textures[i])
            img_data = image.tobytes("raw", "RGBX" if image.mode == 'RGB' else "RGBA", 0, -1)
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, image.size[0], image.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data);
        
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def build_skybox(self):       
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)
            
    def draw(self, viewToClipTransform, worldToViewTransform):
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "skybox"), 0)
        setUniform(self.shader, "viewToClipTransform", viewToClipTransform)
        setUniform(self.shader, "worldToViewTransform", worldToViewTransform)
        
        
        glDepthMask(GL_FALSE)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texId)
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        
        
        glBindVertexArray(0)
        glDepthMask(GL_TRUE) 
        glUseProgram(0)
        
    def generate_vertices(self):
        self.vertices = np.array([
                        -1.0,  1.0, -1.0,
                        -1.0, -1.0, -1.0,
                        1.0, -1.0, -1.0,
                        1.0, -1.0, -1.0,
                        1.0,  1.0, -1.0,
                        -1.0,  1.0, -1.0,

                        -1.0, -1.0,  1.0,
                        -1.0, -1.0, -1.0,
                        -1.0,  1.0, -1.0,
                        -1.0,  1.0, -1.0,
                        -1.0,  1.0,  1.0,
                        -1.0, -1.0,  1.0,

                        1.0, -1.0, -1.0,
                        1.0, -1.0,  1.0,
                        1.0,  1.0,  1.0,
                        1.0,  1.0,  1.0,
                        1.0,  1.0, -1.0,
                        1.0, -1.0, -1.0,

                        -1.0, -1.0,  1.0,
                        -1.0,  1.0,  1.0,
                        1.0,  1.0,  1.0,
                        1.0,  1.0,  1.0,
                        1.0, -1.0,  1.0,
                        -1.0, -1.0,  1.0,

                        -1.0,  1.0, -1.0,
                        1.0,  1.0, -1.0,
                        1.0,  1.0,  1.0,
                        1.0,  1.0,  1.0,
                        -1.0,  1.0,  1.0,
                        -1.0,  1.0, -1.0,

                        -1.0, -1.0, -1.0,
                        -1.0, -1.0,  1.0,
                        1.0, -1.0, -1.0,
                        1.0, -1.0, -1.0,
                        -1.0, -1.0,  1.0,
                        1.0, -1.0,  1.0
                        ], dtype=np.float32)
        
    def generate_shader(self):
        vertexShader = """
        #version 330
        in vec3 positionIn;
        
        uniform mat4 viewToClipTransform;
        uniform mat4 worldToViewTransform;
        
        out vec3 v2f_texCoord;

        void main()
        {
            vec4 viewPos = worldToViewTransform * vec4(positionIn, 1.0);
            gl_Position = viewToClipTransform * vec4(viewPos.xyz, 1.0);
            v2f_texCoord = positionIn;
        }
        """
        fragmentShader = """
        #version 330
        in vec3	v2f_texCoord;
        
        uniform samplerCube skybox;
        
        out vec4 fragmentColor;
        
        void main()
        {    
            fragmentColor = texture(skybox, v2f_texCoord);
        }
        """
        self.shader = buildShader([vertexShader], [fragmentShader], {"positionIn" : 0})
        if self.shader == None:
            raise Exception("Skybox shader not implemented")