from OpenGL.GL import *
import numpy as np
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at
import xml.etree.ElementTree as ET
import math
import imgui
from PIL import Image
from collections import deque

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

# This is a helper class to provide the ability to use * for matrix/matrix and matrix/vector multiplication.
# It also helps out uploading constants and a few other operations as python does not support overloading functions.
# Note that a vector is just represented as a list on floats, and we rely on numpy to take care of the 
class Mat4:
    matData = None
    # Construct a Mat4 from a python array
    def __init__(self, p = [[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]]):
        if isinstance(p, Mat3):
            self.matData = np.matrix(np.identity(4))
            self.matData[:3,:3] = p.matData
        else:
            self.matData = np.matrix(p)

    # overload the multiplication operator to enable sane looking transformation expressions!
    def __mul__(self, other):
        # if it is a list, we let numpy attempt to convert the data
        # we then return it as a list also (the typical use case is 
        # for transforming a vector). Could be made more robust...
        if isinstance(other, (np.ndarray, list)):
            return list(self.matData.dot(other).flat)
        # Otherwise we assume it is another Mat4 or something compatible, and just multiply the matrices
        # and return the result as a new Mat4
        return Mat4(self.matData.dot(other.matData))
    
    # Helper to get data as a contiguous array for upload to OpenGL
    def getData(self):
        return np.ascontiguousarray(self.matData, dtype=np.float32)

    # note: returns an inverted copy, does not change the object (for clarity use the global function instead)
    #       only implemented as a member to make it easy to overload based on matrix class (i.e. 3x3 or 4x4)
    def _inverse(self):
        return Mat4(np.linalg.inv(self.matData))

    def _transpose(self):
        return Mat4(self.matData.T)

    def _set_open_gl_uniform(self, loc):
        glUniformMatrix4fv(loc, 1, GL_TRUE, self.getData())



class Mat3:
    matData = None
    # Construct a Mat4 from a python array
    def __init__(self, p = [[1,0,0],
                            [0,1,0],
                            [0,0,1]]):
        if isinstance(p, Mat4):
            self.matData = p.matData[:3,:3]
        else:
            self.matData = np.matrix(p)

    # overload the multiplication operator to enable sane looking transformation expressions!
    def __mul__(self, other):
        # if it is a list, we let numpy attempt to convert the data
        # we then return it as a list also (the typical use case is 
        # for transforming a vector). Could be made more robust...
        if isinstance(other, (np.ndarray, list)):
            return list(self.matData.dot(other).flat)
        # Otherwise we assume it is another Mat3 or something compatible, and just multiply the matrices
        # and return the result as a new Mat3
        return Mat3(self.matData.dot(other.matData))
    
    # Helper to get data as a contiguous array for upload to OpenGL
    def getData(self):
        return np.ascontiguousarray(self.matData, dtype=np.float32)

    # note: returns an inverted copy, does not change the object (for clarity use the global function instead)
    #       only implemented as a member to make it easy to overload based on matrix class (i.e. 3x3 or 4x4)
    def _inverse(self):
        return Mat3(np.linalg.inv(self.matData))

    def _transpose(self):
        return Mat3(self.matData.T)

    def _set_open_gl_uniform(self, loc):
        glUniformMatrix3fv(loc, 1, GL_TRUE, self.getData())


#
# matrix consruction functions
#

def make_translation(x, y, z):
    return Mat4([[1,0,0,x],
                 [0,1,0,y],
                 [0,0,1,z],
                 [0,0,0,1]])


def make_translation(x, y, z):
    return Mat4([[1,0,0,x],
                 [0,1,0,y],
                 [0,0,1,z],
                 [0,0,0,1]])

 
def make_scale(x, y, z):
    return Mat4([[x,0,0,0],
                 [0,y,0,0],
                 [0,0,z,0],
                 [0,0,0,1]])

def make_uniform_scale(s):
    return make_scale(s,s,s)

def make_rotation_y(angle):
    return Mat4([[math.cos(angle), 0, -math.sin(angle),0],
                 [0,1,0,0],
                 [math.sin(angle),0, math.cos(angle),0],
                 [0,0,0,1]])


def make_rotation_x(angle):
    return Mat4([[1,0,0,0],
                 [0, math.cos(angle), -math.sin(angle),0],
                 [0, math.sin(angle), math.cos(angle),0],
                 [0,0,0,1]])


def make_rotation_z(angle):
    return Mat4([[math.cos(angle),-math.sin(angle),0,0],
                 [math.sin(angle),math.cos(angle),0,0],
                 [0,0,1,0],
                 [0,0,0,1]])


# 
# Matrix operations
#

# note: returns an inverted copy, does not change the object (for clarity use the global function instead)
def inverse(mat):
    return mat._inverse()

def transpose(mat):
    return mat._transpose()


#
# Helper function to set the parameters that the ObjModel implementation expects.
# Most of what happens here is beyond the scope of this lab! 
#
def drawObjModel(viewToClipTfm, worldToViewTfm, modelToWorldTfm, model):
    # Lighting/Shading is very often done in view space, which is why a transformation that lands positions in this space is needed
    modelToViewTransform = worldToViewTfm * modelToWorldTfm
    
    # this is a special transform that ensures that normal vectors remain orthogonal to the 
    # surface they are supposed to be even in the prescence of non-uniform scaling.
    # It is a 3x3 matrix as vectors don't need translation anyway and this transform is only for vectors,
    # not points. If there is no non-uniform scaling this is just the same as Mat3(modelToViewTransform)
    modelToViewNormalTransform = lu.inverse(lu.transpose(lu.Mat3(modelToViewTransform)));

    # Bind the shader program such that we can set the uniforms (model.render sets it again)
    glUseProgram(model.defaultShader)

    # transform (rotate) light direction into view space (as this is what the ObjModel shader wants)
    viewSpaceLightDirection = lu.normalize(lu.Mat3(worldToViewTfm) * g_worldSpaceLightDirection)
    glUniform3fv(glGetUniformLocation(model.defaultShader, "viewSpaceLightDirection"), 1, viewSpaceLightDirection);

    # This dictionary contains a few transforms that are needed to render the ObjModel using the default shader.
    # it would be possible to just set the modelToWorld transform, as this is the only thing that changes between
    # the objects, and compute the other matrices in the vertex shader.
    # However, this would push a lot of redundant computation to the vertex shader and makes the code less self contained,
    # in this way we set all the required parameters explicitly.
    transforms = {
        "modelToClipTransform" : viewToClipTfm * worldToViewTfm * modelToWorldTfm,
        "modelToViewTransform" : modelToViewTransform,
        "modelToViewNormalTransform" : modelToViewNormalTransform,
    }
    
    model.render(None, None, transforms)


#
# vector operations
#

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm


def vec2(x, y = None):
    if y == None:
        return np.array([x,x], dtype=np.float32)
    return np.array([x,y], dtype=np.float32)

def vec3(x, y = None, z = None):
    if y == None:
        return np.array([x,x,x], dtype=np.float32)
    if z == None:
        return np.array([x,y,y], dtype=np.float32)
    return np.array([x, y, z], dtype=np.float32)

# The reason we need a 'look from', and don't just use lookAt(pos, pos+dir, up) is because if pos is large (i.e., far from the origin) and 'dir' is a unit vector (common case)
# then the precision loss in the addition followed by subtraction in lookAt to get the direction back is _significant_, and leads to jerky camera movements.
def make_lookFrom(eye, direction, up):
    f = normalize(direction)
    U = np.array(up[:3])
    s = normalize(np.cross(f, U))
    u = np.cross(s, f)
    M = np.matrix(np.identity(4))
    M[:3,:3] = np.vstack([s,u,-f])
    T = make_translation(-eye[0], -eye[1], -eye[2])
    return Mat4(M) * T

#
# This function performs the steps needed to compile the source code for a
# shader stage (e.g., vertex / fragment) and attach it to a shader program object.
#
def compileAndAttachShader(shaderProgram, shaderType, sources):
    # Create the opengl shader object
    shader = glCreateShader(shaderType)
    # upload the source code for the shader
    # Note the function takes an array of source strings and lengths.
    glShaderSource(shader, sources)
    glCompileShader(shader)

    # If there is a syntax or other compiler error during shader compilation,
    # we'd like to know
    compileOk = glGetShaderiv(shader, GL_COMPILE_STATUS)

    if not compileOk:
        err = getShaderInfoLog(shader)
        print("SHADER COMPILE ERROR: '%s'" % err);
        return False

    glAttachShader(shaderProgram, shader)
    glDeleteShader(shader)
    return True

# creates a shader with a vertex and fragment shader that binds a map of attribute streams 
# to the shader and the also any number of output shader variables
# The fragDataLocs can be left out for programs that don't use multiple render targets as 
# the default for any output variable is zero.
def buildShader(vertexShaderSources, fragmentShaderSources, attribLocs, fragDataLocs = {}):
    shader = glCreateProgram()

    if compileAndAttachShader(shader, GL_VERTEX_SHADER, vertexShaderSources) and compileAndAttachShader(shader, GL_FRAGMENT_SHADER, fragmentShaderSources):
	    # Link the attribute names we used in the vertex shader to the integer index
        for name, loc in attribLocs.items():
            glBindAttribLocation(shader, loc, name)

	    # If we have multiple images bound as render targets, we need to specify which
	    # 'out' variable in the fragment shader goes where in this case it is totally redundant 
        # as we only have one (the default render target, or frame buffer) and the default binding is always zero.
        for name, loc in fragDataLocs.items():
            glBindFragDataLocation(shader, loc, name)

        # once the bindings are done we can link the program stages to get a complete shader pipeline.
        # this can yield errors, for example if the vertex and fragment shaders don't have compatible out and in 
        # variables (e.g., the fragment shader expects some data that the vertex shader is not outputting).
        glLinkProgram(shader)
        linkStatus = glGetProgramiv(shader, GL_LINK_STATUS)
        if not linkStatus:
            err = glGetProgramInfoLog(shader).decode()
            print("SHADER LINKER ERROR: '%s'" % err)
            glDeleteProgram(shader)
            return None
        return shader
    else:
        glDeleteProgram(shader)
        return None


# Helper for debugging, if uniforms appear to not be set properly, you can set a breakpoint here, 
# or uncomment the printing code. If the 'loc' returned is -1, then the variable is either not 
# declared at all in the shader or it is not used  and therefore removed by the optimizing shader compiler.
def getUniformLocationDebug(shaderProgram, name):
    loc = glGetUniformLocation(shaderProgram, name)
    # Useful point for debugging, replace with silencable logging 
    # TODO: should perhaps replace this with the standard python logging facilities
    #if loc == -1:
    #    print("Uniforn '%s' was not found"%name)
    return loc

def getShaderInfoLog(obj):
    logLength = glGetShaderiv(obj, GL_INFO_LOG_LENGTH)

    if logLength > 0:
        return glGetShaderInfoLog(obj).decode()

    return ""

# Helper to set uniforms of different types, looks the way it does since Python does not have support for 
# function overloading (as C++ has for example). This function covers the types used in the code here, but 
# makes no claim of completeness. The last case is for Mat3/Mat4 (above), and if you get an exception 
# on that line, it is likely because the function was cal
def setUniform(shaderProgram, uniformName, value):
    loc = getUniformLocationDebug(shaderProgram, uniformName)
    if isinstance(value, float):
        glUniform1f(loc, value)
    elif isinstance(value, int):
        glUniform1i(loc, value)
    elif isinstance(value, (np.ndarray, list)):
        if len(value) == 2:
            glUniform2fv(loc, 1, value)
        if len(value) == 3:
            glUniform3fv(loc, 1, value)
        if len(value) == 4:
            glUniform4fv(loc, 1, value)
    elif isinstance(value, (Mat3, Mat4)):
        value._set_open_gl_uniform(loc)
    else:
        assert False # If this happens the type was not supported, check your argument types and either add a new else case above or change the type


# Recursively subdivide a triangle with its vertices on the surface of the unit sphere such that the new vertices also are on part of the unit sphere.
def subDivide(dest, v0, v1, v2, level):
	#If the level index/counter is non-zero...
	if level:
		# ...we subdivide the input triangle into four equal sub-triangles
		# The mid points are the half way between to vertices, which is really (v0 + v2) / 2, but 
		# instead we normalize the vertex to 'push' it out to the surface of the unit sphere.
		v3 = normalize(v0 + v1);
		v4 = normalize(v1 + v2);
		v5 = normalize(v2 + v0);

		# ...and then recursively call this function for each of those (with the level decreased by one)
		subDivide(dest, v0, v3, v5, level - 1);
		subDivide(dest, v3, v4, v5, level - 1);
		subDivide(dest, v3, v1, v4, level - 1);
		subDivide(dest, v5, v4, v2, level - 1);
	else:
		# If we have reached the terminating level, just output the vertex position
		dest.append(v0)
		dest.append(v1)
		dest.append(v2)


# Turns a multidimensional array (up to 3d?) into a 1D array
def flatten(*lll):
	return [u for ll in lll for l in ll for u in l]

def uploadFloatData(bufferObject, floatData):
    flatData = flatten(floatData)
    data_buffer = (c_float * len(flatData))(*flatData)
    # Upload data to the currently bound GL_ARRAY_BUFFER, note that this is
    # completely anonymous binary data, no type information is retained (we'll
    # supply that later in glVertexAttribPointer)
    glBindBuffer(GL_ARRAY_BUFFER, bufferObject)
    glBufferData(GL_ARRAY_BUFFER, data_buffer, GL_STATIC_DRAW)

def createVertexArrayObject():
	return glGenVertexArrays(1);

def createAndAddVertexArrayData(vertexArrayObject, data, attributeIndex):
    glBindVertexArray(vertexArrayObject)
    buffer = glGenBuffers(1)
    uploadFloatData(buffer, data)

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glVertexAttribPointer(attributeIndex, len(data[0]), GL_FLOAT, GL_FALSE, 0, None);
    glEnableVertexAttribArray(attributeIndex);

    # Unbind the buffers again to avoid unintentianal GL state corruption (this is something that can be rather inconventient to debug)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return buffer

def createAndAddIndexArray(vertexArrayObject, indexData):
    glBindVertexArray(vertexArrayObject);
    indexBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, indexBuffer);

    data_buffer = (c_uint * len(indexData))(*indexData)
    glBufferData(GL_ARRAY_BUFFER, data_buffer, GL_STATIC_DRAW);

    # Bind the index buffer as the element array buffer of the VAO - this causes it to stay bound to this VAO - fairly unobvious.
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

    # Unbind the buffers again to avoid unintentianal GL state corruption (this is something that can be rather inconventient to debug)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return indexBuffer;


def createSphere(numSubDivisionLevels):
	sphereVerts = []

	# The root level sphere is formed from 8 triangles in a diamond shape (two pyramids)
	subDivide(sphereVerts, vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 0, 0), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, 1, 0), vec3(1, 0, 0), vec3(0, 0, -1), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, 1, 0), vec3(0, 0, -1), vec3(-1, 0, 0), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, 1, 0), vec3(-1, 0, 0), vec3(0, 0, 1), numSubDivisionLevels)

	subDivide(sphereVerts, vec3(0, -1, 0), vec3(1, 0, 0), vec3(0, 0, 1), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, -1, 0), vec3(0, 0, 1), vec3(-1, 0, 0), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, -1, 0), vec3(-1, 0, 0), vec3(0, 0, -1), numSubDivisionLevels)
	subDivide(sphereVerts, vec3(0, -1, 0), vec3(0, 0, -1), vec3(1, 0, 0), numSubDivisionLevels)

	return sphereVerts;


g_sphereVertexArrayObject = None
g_sphereShader = None
g_numSphereVerts = 0

def drawSphere(position, radius, sphereColour, viewToClipTransform, worldToViewTransform):
    global g_sphereVertexArrayObject
    global g_sphereShader
    global g_numSphereVerts

    modelToWorldTransform = make_translation(position[0], position[1], position[2]) * make_scale(radius, radius, radius);

    if not g_sphereVertexArrayObject:
        sphereVerts = createSphere(3)
        g_numSphereVerts = len(sphereVerts)
        g_sphereVertexArrayObject = createVertexArrayObject()
        createAndAddVertexArrayData(g_sphereVertexArrayObject, sphereVerts, 0)
        # redundantly add as normals...
        createAndAddVertexArrayData(g_sphereVertexArrayObject, sphereVerts, 1)



        vertexShader = """
            #version 330
            in vec3 positionIn;
            in vec3 normalIn;

            uniform mat4 modelToClipTransform;
            uniform mat4 modelToViewTransform;
            uniform mat3 modelToViewNormalTransform;

            // 'out' variables declared in a vertex shader can be accessed in the subsequent stages.
            // For a fragment shader the variable is interpolated (the type of interpolation can be modified, try placing 'flat' in front here and in the fragment shader!).
            out VertexData
            {
                vec3 v2f_viewSpacePosition;
                vec3 v2f_viewSpaceNormal;
            };

            void main() 
            {
                v2f_viewSpacePosition = (modelToViewTransform * vec4(positionIn, 1.0)).xyz;
                v2f_viewSpaceNormal = normalize(modelToViewNormalTransform * normalIn);

	            // gl_Position is a buit-in 'out'-variable that gets passed on to the clipping and rasterization stages (hardware fixed function).
                // it must be written by the vertex shader in order to produce any drawn geometry. 
                // We transform the position using one matrix multiply from model to clip space. Note the added 1 at the end of the position to make the 3D
                // coordinate homogeneous.
	            gl_Position = modelToClipTransform * vec4(positionIn, 1.0);
            }
"""

        fragmentShader = """
            #version 330
            // Input from the vertex shader, will contain the interpolated (i.e., area weighted average) vaule out put for each of the three vertex shaders that 
            // produced the vertex data for the triangle this fragmet is part of.
            in VertexData
            {
                vec3 v2f_viewSpacePosition;
                vec3 v2f_viewSpaceNormal;
            };

            uniform vec4 sphereColour;

            out vec4 fragmentColor;

            void main() 
            {
                float shading = max(0.0, dot(normalize(-v2f_viewSpacePosition), v2f_viewSpaceNormal));
	            fragmentColor = vec4(sphereColour.xyz * shading, sphereColour.w);

            }
"""
        g_sphereShader = buildShader([vertexShader], [fragmentShader], {"positionIn" : 0, "normalIn" : 1})


    glUseProgram(g_sphereShader)
    setUniform(g_sphereShader, "sphereColour", sphereColour)

    modelToClipTransform = viewToClipTransform * worldToViewTransform * modelToWorldTransform
    modelToViewTransform = worldToViewTransform * modelToWorldTransform
    modelToViewNormalTransform = inverse(transpose(Mat3(modelToViewTransform)))
    setUniform(g_sphereShader, "modelToClipTransform", modelToClipTransform);
    setUniform(g_sphereShader, "modelToViewTransform", modelToViewTransform);
    setUniform(g_sphereShader, "modelToViewNormalTransform", modelToViewNormalTransform);


    glBindVertexArray(g_sphereVertexArrayObject)
    glDrawArrays(GL_TRIANGLES, 0, g_numSphereVerts)



# Helper function to extend a 3D point to homogeneous, transform it and back again.
# (For practically all cases except projection, the W component is still 1 after,
# but this covers the correct implementation).
# Note that it does not work for vectors! For vectors we're usually better off just using the 3x3 part of the matrix.
def transformPoint(mat4x4, point):
    x,y,z,w = mat4x4 * [point[0], point[1], point[2], 1.0]
    return vec3(x,y,z) / w



def imguiX_color_edit3_list(label, v):
    a,b = imgui.color_edit3(label, *v)#, imgui.GuiColorEditFlags_Float);// | ImGuiColorEditFlags_HSV);
    return a,list(b)


# make_lookAt defines a view transform, i.e., from world to view space, using intuitive parameters. location of camera, point to aim, and rough up direction.
# this is basically the same as what we saw in Lexcture #2 for placing the car in the world, except the inverse! (and also view-space 'forwards' is the negative z-axis)
def make_lookAt(eye, target, up):
    return make_lookFrom(eye, np.array(target[:3]) - np.array(eye[:3]), up)



def make_perspective(fovy, aspect, n, f):
    radFovY = math.radians(fovy)
    tanHalfFovY = math.tan(radFovY / 2.0)
    sx = 1.0 / (tanHalfFovY * aspect)
    sy = 1.0 / tanHalfFovY
    zz = -(f + n) / (f - n)
    zw = -(2.0 * f * n) / (f - n)

    return Mat4([[sx,0,0,0],
                 [0,sy,0,0],
                 [0,0,zz,zw],
                 [0,0,-1,0]])


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
        cameraDirection = Mat3(make_rotation_y(math.radians(self.yawDeg))) * Mat3(make_rotation_x(math.radians(self.pitchDeg))) * [0,0,1]
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
                nx, ny, nz = x / self.radius, y / self.radius, z / self.radius # Normals are same as position for a unit sphere
                s = j / self.sectorCount
                t = i / self.stackCount
                self.sphere_vertices.extend([x, y, z, nx, ny, nz, s, t]) # Append normals and texture coordinates to vertices


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

class TextFont:
    def __init__(self):
        self.shader = None
        self.load_font('data/fonts/sans-serif.png', 'data/fonts/sans-serif.xml')

    def load_font(self, image_path, xml_path):
        # Load font atlas texture
        image = Image.open(image_path).convert('RGBA')
        width, height = image.size
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, np.array(image))
        
        # Load glyph metrics from XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.glyphs = {g.attrib['char']: g.attrib for g in root.iter('glyph')}
        
    def draw(self, text, toClipTransform, color):
        # Compile shader program if not done yet
        if self.shader is None:
            self.shader = self.compile_shader()
        
        # Activate shader program
        glUseProgram(self.shader)
        
        # Set uniform variables
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "toClipTransform"), 1, GL_TRUE, toClipTransform)
        glUniform4f(glGetUniformLocation(self.shader, "color"), *color)
        
        # Render glyphs
        glBindTexture(GL_TEXTURE_2D, self.texture)
        for c in text:
            g = self.glyphs.get(c)
            if g is not None:
                x, y, w, h = map(int, (g['x'], g['y'], g['width'], g['height']))
                
                glBegin(GL_QUADS)
                glTexCoord2f(x, y); glVertex2f(x, y)
                glTexCoord2f(x+w, y); glVertex2f(x+w, y)
                glTexCoord2f(x+w, y+h); glVertex2f(x+w, y+h)
                glTexCoord2f(x, y+h); glVertex2f(x, y+h)
                glEnd()
        
    def compile_shader(self):
        # Shader program that renders a textured quad with uniform color
        vertex_shader_source = """
        #version 330 core
        in vec2 aPos;
        in vec2 aTexCoord;
        out vec2 TexCoord;
        uniform mat4 toClipTransform;
        void main() {
            gl_Position = toClipTransform * vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }"""
        
        fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;
        uniform sampler2D text;
        uniform vec4 color;
        void main() {
            FragColor = texture(text, TexCoord) * color;
        }"""
        
        # Compile vertex shader
        self.shader = buildShader([vertex_shader_source], [fragment_shader_source], {"aPos" : 0, "aTexCoord" : 1})
        
       

       



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
        
