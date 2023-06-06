from OpenGL.GL import *
import numpy as np
from ctypes import sizeof, c_float, c_void_p, c_uint, string_at
import xml.etree.ElementTree as ET
import math
import imgui
import freetype
from noise import snoise3
from OpenGL.GL import *
import glfw, sys, imgui, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from imgui.integrations.glfw import GlfwRenderer as ImGuiGlfwRenderer


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

def vec4(x, y = None, z = None, w = None):
    if y == None:
        return np.array([x,x,x,x], dtype=np.float32)
    if z == None:
        return np.array([x,y,y,y], dtype=np.float32)
    if w == None:
        return np.array([x,y,z,z], dtype=np.float32)
    return np.array([x, y, z, w], dtype=np.float32)

def dot(a, b):
    return np.dot(a, b)


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



g_mousePos = [0.0, 0.0]

VAL_Position = 0
g_vertexDataBuffer = 0
g_vertexArrayObject = 0
g_simpleShader = 0


def beginImGuiHud():
    global g_mousePos
    imgui.set_next_window_position(5.0, 5.0)

    if imgui.begin("Example: Fixed Overlay", 0, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_FOCUS_ON_APPEARING):
        pass


def endImGuiHud():
        imgui.end()


def runProgram(title, startWidth, startHeight, renderFrame, initResources = None, drawUi = None, update = None):
    global g_simpleShader
    global g_vertexArrayObject
    global g_vertexDataBuffer
    global g_mousePos

    if not glfw.init():
        sys.exit(1)

    #glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, 1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)


    window = glfw.create_window(startWidth, startHeight, title, None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window) 

    print("--------------------------------------\nOpenGL\n  Vendor: %s\n  Renderer: %s\n  Version: %s\n--------------------------------------\n" % (glGetString(GL_VENDOR).decode("utf8"), glGetString(GL_RENDERER).decode("utf8"), glGetString(GL_VERSION).decode("utf8")), flush=True)
    imgui.create_context()
    impl = ImGuiGlfwRenderer(window)

    
    glDisable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    if initResources:
        initResources()
        
    currentTime = glfw.get_time()
    prevMouseX,prevMouseY = glfw.get_cursor_pos(window)

    while not glfw.window_should_close(window):
        prevTime = currentTime
        currentTime = glfw.get_time()
        dt = currentTime - prevTime

        keyStateMap = {}
        for name,id in g_glfwKeymap.items():
            keyStateMap[name] = glfw.get_key(window, id) == glfw.PRESS

        for name,id in g_glfwMouseMap.items():
            keyStateMap[name] = glfw.get_mouse_button(window, id) == glfw.PRESS

        mouseX,mouseY = glfw.get_cursor_pos(window)
        g_mousePos = [mouseX,mouseY]

        imIo = imgui.get_io()
        mouseDelta = [mouseX - prevMouseX, mouseY - prevMouseY]
        if imIo.want_capture_mouse:
            mouseDelta = [0,0]
        update(dt, keyStateMap, mouseDelta)
        prevMouseX,prevMouseY = mouseX,mouseY
        
        # Render here, e.g.  using pyOpenGL
        width, height = glfw.get_framebuffer_size(window)

        imgui.new_frame()

        beginImGuiHud()

        renderFrame(width, height)
    
        if drawUi:
            drawUi()

        endImGuiHud()
        imgui.render()
        impl.render(imgui.get_draw_data())
        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()
        impl.process_inputs()

    glfw.terminate()



g_glfwMouseMap = {
    "MOUSE_BUTTON_LEFT" : glfw.MOUSE_BUTTON_LEFT,
    "MOUSE_BUTTON_RIGHT" : glfw.MOUSE_BUTTON_RIGHT,
    "MOUSE_BUTTON_MIDDLE" : glfw.MOUSE_BUTTON_MIDDLE,
}


g_glfwKeymap = {
    "SPACE" : glfw.KEY_SPACE,
    "APOSTROPHE" : glfw.KEY_APOSTROPHE,
    "COMMA" : glfw.KEY_COMMA,
    "MINUS" : glfw.KEY_MINUS,
    "PERIOD" : glfw.KEY_PERIOD,
    "SLASH" : glfw.KEY_SLASH,
    "0" : glfw.KEY_0,
    "1" : glfw.KEY_1,
    "2" : glfw.KEY_2,
    "3" : glfw.KEY_3,
    "4" : glfw.KEY_4,
    "5" : glfw.KEY_5,
    "6" : glfw.KEY_6,
    "7" : glfw.KEY_7,
    "8" : glfw.KEY_8,
    "9" : glfw.KEY_9,
    "SEMICOLON" : glfw.KEY_SEMICOLON,
    "EQUAL" : glfw.KEY_EQUAL,
    "A" : glfw.KEY_A,
    "B" : glfw.KEY_B,
    "C" : glfw.KEY_C,
    "D" : glfw.KEY_D,
    "E" : glfw.KEY_E,
    "F" : glfw.KEY_F,
    "G" : glfw.KEY_G,
    "H" : glfw.KEY_H,
    "I" : glfw.KEY_I,
    "J" : glfw.KEY_J,
    "K" : glfw.KEY_K,
    "L" : glfw.KEY_L,
    "M" : glfw.KEY_M,
    "N" : glfw.KEY_N,
    "O" : glfw.KEY_O,
    "P" : glfw.KEY_P,
    "Q" : glfw.KEY_Q,
    "R" : glfw.KEY_R,
    "S" : glfw.KEY_S,
    "T" : glfw.KEY_T,
    "U" : glfw.KEY_U,
    "V" : glfw.KEY_V,
    "W" : glfw.KEY_W,
    "X" : glfw.KEY_X,
    "Y" : glfw.KEY_Y,
    "Z" : glfw.KEY_Z,
    "LEFT_BRACKET" : glfw.KEY_LEFT_BRACKET,
    "BACKSLASH" : glfw.KEY_BACKSLASH,
    "RIGHT_BRACKET" : glfw.KEY_RIGHT_BRACKET,
    "GRAVE_ACCENT" : glfw.KEY_GRAVE_ACCENT,
    "WORLD_1" : glfw.KEY_WORLD_1,
    "WORLD_2" : glfw.KEY_WORLD_2,
    "ESCAPE" : glfw.KEY_ESCAPE,
    "ENTER" : glfw.KEY_ENTER,
    "TAB" : glfw.KEY_TAB,
    "BACKSPACE" : glfw.KEY_BACKSPACE,
    "INSERT" : glfw.KEY_INSERT,
    "DELETE" : glfw.KEY_DELETE,
    "RIGHT" : glfw.KEY_RIGHT,
    "LEFT" : glfw.KEY_LEFT,
    "DOWN" : glfw.KEY_DOWN,
    "UP" : glfw.KEY_UP,
    "PAGE_UP" : glfw.KEY_PAGE_UP,
    "PAGE_DOWN" : glfw.KEY_PAGE_DOWN,
    "HOME" : glfw.KEY_HOME,
    "END" : glfw.KEY_END,
    "CAPS_LOCK" : glfw.KEY_CAPS_LOCK,
    "SCROLL_LOCK" : glfw.KEY_SCROLL_LOCK,
    "NUM_LOCK" : glfw.KEY_NUM_LOCK,
    "PRINT_SCREEN" : glfw.KEY_PRINT_SCREEN,
    "PAUSE" : glfw.KEY_PAUSE,
    "F1" : glfw.KEY_F1,
    "F2" : glfw.KEY_F2,
    "F3" : glfw.KEY_F3,
    "F4" : glfw.KEY_F4,
    "F5" : glfw.KEY_F5,
    "F6" : glfw.KEY_F6,
    "F7" : glfw.KEY_F7,
    "F8" : glfw.KEY_F8,
    "F9" : glfw.KEY_F9,
    "F10" : glfw.KEY_F10,
    "F11" : glfw.KEY_F11,
    "F12" : glfw.KEY_F12,
    "F13" : glfw.KEY_F13,
    "F14" : glfw.KEY_F14,
    "F15" : glfw.KEY_F15,
    "F16" : glfw.KEY_F16,
    "F17" : glfw.KEY_F17,
    "F18" : glfw.KEY_F18,
    "F19" : glfw.KEY_F19,
    "F20" : glfw.KEY_F20,
    "F21" : glfw.KEY_F21,
    "F22" : glfw.KEY_F22,
    "F23" : glfw.KEY_F23,
    "F24" : glfw.KEY_F24,
    "F25" : glfw.KEY_F25,
    "KP_0" : glfw.KEY_KP_0,
    "KP_1" : glfw.KEY_KP_1,
    "KP_2" : glfw.KEY_KP_2,
    "KP_3" : glfw.KEY_KP_3,
    "KP_4" : glfw.KEY_KP_4,
    "KP_5" : glfw.KEY_KP_5,
    "KP_6" : glfw.KEY_KP_6,
    "KP_7" : glfw.KEY_KP_7,
    "KP_8" : glfw.KEY_KP_8,
    "KP_9" : glfw.KEY_KP_9,
    "KP_DECIMAL" : glfw.KEY_KP_DECIMAL,
    "KP_DIVIDE" : glfw.KEY_KP_DIVIDE,
    "KP_MULTIPLY" : glfw.KEY_KP_MULTIPLY,
    "KP_SUBTRACT" : glfw.KEY_KP_SUBTRACT,
    "KP_ADD" : glfw.KEY_KP_ADD,
    "KP_ENTER" : glfw.KEY_KP_ENTER,
    "KP_EQUAL" : glfw.KEY_KP_EQUAL,
    "LEFT_SHIFT" : glfw.KEY_LEFT_SHIFT,
    "LEFT_CONTROL" : glfw.KEY_LEFT_CONTROL,
    "LEFT_ALT" : glfw.KEY_LEFT_ALT,
    "LEFT_SUPER" : glfw.KEY_LEFT_SUPER,
    "RIGHT_SHIFT" : glfw.KEY_RIGHT_SHIFT,
    "RIGHT_CONTROL" : glfw.KEY_RIGHT_CONTROL,
    "RIGHT_ALT" : glfw.KEY_RIGHT_ALT,
    "RIGHT_SUPER" : glfw.KEY_RIGHT_SUPER,
    "MENU" : glfw.KEY_MENU,
}
    

class CharacterSlot:
    def __init__(self, texture, glyph):
        self.texture = texture
        self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)

        if isinstance(glyph, freetype.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
        elif isinstance(glyph, freetype.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
        else:
            raise RuntimeError('unknown glyph type')


class Text:
    def __init__(self, fontfile):
        self.fontfile = fontfile
        self.characters = dict()
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)

        self.generate_buffers()
        self.generate_shader()

    def generate_buffers(self):
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
        
        glEnableVertexAttribArray(0)       
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        face = freetype.Face(self.fontfile)
        face.set_char_size( 48*64 )

        #load first 128 characters of ASCII set
        for i in range(0,128):
            character = chr(i)
            face.load_char(character)
            glyph = face.glyph
            
            #generate texture
            textId = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, textId)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, glyph.bitmap.width, glyph.bitmap.rows, 0,
                        GL_RED, GL_UNSIGNED_BYTE, glyph.bitmap.buffer)

            #texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            #now store character for later use
            self.characters[chr(i)] = CharacterSlot(textId, glyph)

        glBindTexture(GL_TEXTURE_2D, 0)
        
    def generate_vertices(self, xpos, ypos, w, h):
        return np.asarray([
            xpos,     ypos - h, 0, 0,
            xpos,     ypos,     0, 1,
            xpos + w, ypos,     1, 1,
            xpos,     ypos - h, 0, 0,
            xpos + w, ypos,     1, 1,
            xpos + w, ypos - h, 1, 0
        ], np.float32)

    def render_text(self, text, modelToClipTransform, color, scale):
        glUseProgram(self.shader)
        face = freetype.Face(self.fontfile)
        face.set_char_size(48*64)

        setUniform(self.shader, "modelToClipTransform", modelToClipTransform)
        setUniform(self.shader, "textColor", color)

        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindVertexArray(self.VAO)

        x, y = 1, 1
        for c in text:            
            ch = self.characters[c]
            
            w, h = ch.textureSize
            # Adjust x and y based on the character bearing
            w = w * scale
            h = h * scale

            vertices = self.generate_vertices(x, y, w, h)


            #render glyph texture over quad
            glBindTexture(GL_TEXTURE_2D, ch.texture)
            #update content of VBO memory
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            #render quad
            glDrawArrays(GL_TRIANGLES, 0, 6)
            #now advance cursors for next glyph (note that advance is number of 1/64 pixels)
            x += (ch.advance>>6)*scale

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def generate_shader(self):
        vertexShader = """
        #version 330
        in vec4 positionIn;
        
        uniform mat4 modelToClipTransform;

        out vec2 v2f_texCoord;

        void main()
        {
            gl_Position = modelToClipTransform * vec4(positionIn.xy, 0.0, 1.0);
            v2f_texCoord = positionIn.zw;
        }
        """
        fragmentShader = """
        #version 330
        in vec2 v2f_texCoord;
    
        uniform sampler2D texture1;
        uniform vec3 textColor;
        
        out vec4 fragmentColor;

        void main()
        {    
            vec4 sampled = vec4(1.0, 1.0, 1.0, texture(texture1, v2f_texCoord).r);
            fragmentColor = vec4(textColor, 1.0) * sampled;
        }
        """
        self.shader = buildShader([vertexShader], [fragmentShader], {"positionIn" : 0})
