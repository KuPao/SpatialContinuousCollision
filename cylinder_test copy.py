from operator import ne
from platform import dist
from mpmath.functions.rszeta import coef
from sympy.core import symbol
from sympy.ntheory.residue_ntheory import _discrete_log_trial_mul
from sympy.solvers.diophantine.diophantine import length
import taichi as ti
import tina
import math
import numpy as np
from sympy import *
import time
from datetime import timedelta
from scipy.optimize import minimize, fsolve, broyden1, root
import autograd.numpy as np_grad
from autograd import grad

ti.init(ti.gpu)

gravity = 9.8
frac = 1
time_k = 1
dt = 5e-4 / time_k
steps = 30
angle = 90 / 180 * math.pi
rot_matrix = tina.eularXYZ([0, 0, 0])
N = 2
NN = N, N
W = 30
L = W / N
x, y, z, t = symbols('x,y,z,t')
#plane = -1 * x + 2 * y
plane = x*x - 14*y
friction = 0.5
theta = 0.0
is_collision = 0
surface_point = []
direction = 1
distance = 0
next_colli_p = np.empty((0,3), np.float32)
next_colli_t = 1
collision_trace = np.empty((0,3), np.float32)
new_p = np.array([0,0,0])
new_v = np.array([0,0,0])
inited = 0
fra = -1
coeff_xx = 0.0
coeff_yy = 0.0
coeff_zz = 0.0
coeff_xy = 0.0
coeff_yz = 0.0
coeff_zx = 0.0
coeff_x = 0.0
coeff_y = 0.0
coeff_z = 0.0
constant = 0.0
coeff_a = 0.0
coeff_b = 0.0
coeff_c = 0.0
coeff_d = 0.0
point = []
dfdL = 0

scene = tina.Scene(smoothing=True, taa=False, ibl=True)

roughness = tina.Param(float, initial=0)#.15)
metallic = tina.Param(float, initial=0)#.25)
material = tina.PBR(metallic=metallic, roughness=roughness)

gui = ti.GUI('cylinder')
# roughness.make_slider(gui, 'roughness')
# metallic.make_slider(gui, 'metallic')

#scene.init_control(gui, theta=0.21066669486962802, phi=0.4295146206079795, radius = 16.853932584269664)
scene.init_control(gui, theta=0., phi=0., radius = 16.853932584269664)

#model = tina.PrimitiveMesh.sphere()

model = tina.MeshTransform(tina.PrimitiveMesh.cylinder(lats=1,rad=0.5,hei=1))

# transform the mesh using the `tina.Transform` node:
tmodel = tina.MeshTransform(model)

surface = tina.MeshNoCulling(tina.MeshGrid((N, N)))
cloth = tina.PBR(basecolor=tina.ChessboardTexture(size=0.2))

line = tina.SimpleLine()
predict_line = tina.SimpleLine()

scene.add_object(line)
scene.add_object(predict_line)
scene.add_object(tmodel, material)
scene.add_object(surface, cloth)

p = ti.Vector.field(3, float, 1)
v = ti.Vector.field(3, float, 1)
acc = ti.Vector.field(3, float, 1)
normal = ti.Vector.field(3, float, 1)
surface_x = ti.Vector.field(3, float, NN)

def objective(X):
    global point
    x, y, z = X
    return (x-point[0])**2 + (y-point[1])**2 + (z-point[2])**2

def plane_eq(X):
    global coeff_xx, coeff_yy, coeff_zz
    global coeff_xy, coeff_yz, coeff_zx
    global coeff_x, coeff_y, coeff_z, constant
    x, y, z = X
    return coeff_xx*x*x+coeff_yy*y*y+coeff_zz*z*z+coeff_xy*x*y+coeff_yz*y*z+coeff_zx*z*x+coeff_x*x+coeff_y*y+coeff_z*z+constant

def line_eq(X):
    global coeff_a, coeff_b, coeff_c, coeff_d
    x, y, z = X
    return coeff_a*x+coeff_b*y+coeff_c*z+coeff_d

def plane_Lagrange(L):
    'Augmented Lagrange function'
    x, y, z, _lambda = L
    return objective([x, y, z]) - _lambda * plane_eq([x, y, z])

def line_Lagrange(L):
    x, y, z, _lambda0, _lanbda1 = L
    return objective([x, y, z]) - _lambda0 * plane_eq([x, y, z]) - _lanbda1 * line_eq([x, y, z])

# Find L that returns all zeros in this function.
def plane_obj(L):
    global dfdL
    x, y, z, _lambda = L
    dFdx, dFdy, dFdz, dFdlam = dfdL(L)
    return [dFdx, dFdy, dFdz, plane_eq([x, y, z])]

def line_obj(L):
    global dfdL
    x, y, z, _lambda0, _lambda1 = L
    dFdx, dFdy, dFdz, dFdlam0, dFdlam1 = dfdL(L)
    return [dFdx, dFdy, dFdz, plane_eq([x, y, z]), line_eq([x, y, z])]


@ti.kernel
def init():
    for i in ti.grouped(surface_x):
        m, n = (i + 0.5) * L - 0.5 - W/2
        surface_x[i] = ti.Vector([m, 0.5*m, n])
        # print(surface_x[i])
    for i in ti.grouped(p):
        #p[i] = ti.Vector([7 - 0.5 / math.sqrt(5), 3.5 + 1 / math.sqrt(5), 0.0])
        p[i] = ti.Vector([7 - 0.5 / math.sqrt(2), 3.5 + 0.5 / math.sqrt(2), 0.0])
        #p[i] = ti.Vector([7 - 0.5 / math.sqrt(5), 4 + 1 / math.sqrt(5), 0.0])
        #x[i] = ti.Vector([7, 3.5, 0.0])
        v[i] = ti.Vector([0.0, 0.0, 0.0])



@ti.kernel
def substep(g: ti.f32, mass: ti.f32, is_collision: ti.i32, normal: ti.ext_arr(), dis: ti.f32, dir:ti.f32, new_p: ti.ext_arr(), new_v: ti.ext_arr()):
    for i in ti.grouped(p):
        p[i] = ti.Vector([new_p[0],new_p[1],new_p[2]])
        v[i] = ti.Vector([new_v[0],new_v[1],new_v[2]])
        acc[i] = g * mass * ti.Vector([0.0, -1, 0.0])
        if is_collision == 1:
            acc[i] += 3200 * mass * dir * dis * ti.Vector([normal[0], normal[1], normal[2]])
        acc[i] /= mass
        v[i] += acc[i] * dt
        p[i] += dt * v[i]

def vector_angle(vec):
    ground = np.array([vec[0], 0, vec[2]])
    l = np.linalg.norm(vec)
    if l < 0.0000001:
        return 0.0
    return acos(np.inner(vec,ground) / np.linalg.norm(vec)/np.linalg.norm(ground))

def normal_plane_curve(plane, normal_plane, v):
    if abs(v[0]) > abs(v[1]) >= abs(v[2]):
        return solve([normal_plane, plane], [x,y,z])
    elif abs(v[0]) > abs(v[2]) >= abs(v[1]):
        return solve([normal_plane, plane], [x,z,y])
    elif abs(v[1]) > abs(v[0]) >= abs(v[2]):
        return solve([normal_plane, plane], [y,x,z])
    elif abs(v[1]) > abs(v[2]) >= abs(v[0]):
        return solve([normal_plane, plane], [y,z,x])
    elif abs(v[2]) > abs(v[0]) >= abs(v[1]):
        return solve([normal_plane, plane], [z,x,y])
    else:
        return solve([normal_plane, plane], [z,y,x])

def curvature_collision_point(o, first, second, vec):
    x_v = vec[0]
    z_v = vec[2]
    a = x_v*x_v + z_v*z_v
    b = 2*(x_v*(second[0]-o[0])+z_v*(second[2]-o[2]))
    c = -((o[0]-first[0])**2+(o[1]-first[1])**2+(o[2]-first[2])**2-(o[1]-second[1])**2-o[0]**2-o[2]**2-second[0]**2-second[2]**2+2*o[0]*second[0]+2*o[1]*second[1])
    d = sqrt(b*b-4*a*c)
    t1 = (-b + d)/2*a
    t2 = (-b - d)/2*a
    if abs(t1) < abs(t2):
        return np.array([second[0]+vec[0]*t1, second[1], second[2]+vec[2]*t1])
    else:
        return np.array([second[0]+vec[0]*t2, second[1], second[2]+vec[2]*t2])

def tina_substep():
    global is_collision, formula
    global numpy_normal, surface_point, direction, distance
    global collision_trace, next_colli_p, next_colli_t
    global p, v, x, y, z, t, plane, dt
    global gui
    global new_p, new_v, inited, fra
    global coeff_xx, coeff_yy, coeff_zz
    global coeff_xy, coeff_yz, coeff_zx
    global coeff_x, coeff_y, coeff_z, constant
    global coeff_a, coeff_b, coeff_c, coeff_d
    global point, dfdL
    collision_point = []
    # return

    if inited == 1:
        new_p = p.to_numpy()[0]
        new_v = v.to_numpy()[0]
    else:
        #new_p = np.array([7 - 0.5 / math.sqrt(5), 3.5 + 1 / math.sqrt(5), 0.0]) + p.to_numpy()[0]
        new_p = np.array([7 - 0.5 / math.sqrt(2), 3.5 + 0.5 / math.sqrt(2), 0.0]) + p.to_numpy()[0]
        new_v = np.array([0,0,0]) + v.to_numpy()[0]
        inited = 1
        return
    if is_collision == 0:
        formula = tina.PrimitiveFormulation(tmodel, 10)
        collision = formula.Calculate_Collision(plane)
        for j in range(3):
            sum = 0
            for i in range(2):
                midpoint = collision[i][j].subs(x, formula.midpoint[0]).subs(y, formula.midpoint[1]).subs(z, formula.midpoint[2])
                if not midpoint.is_real:
                    F = float(formula.mass * gravity)
                    is_collision = 0
                    return
                sum += midpoint
            sum /= 2
            collision_point.append(float(sum))
        numpy_normal = np.array([-2*collision_point[0], 14, 0])
        numpy_normal = numpy_normal / np.linalg.norm(numpy_normal)
        is_collision = 1
        if not any(np.equal(collision_trace, collision_point).all(1)):
            collision_trace = np.append(collision_trace, np.array([collision_point]), axis=0)
            line.set_lines(collision_trace)
        surface_point, direction, distance = formula.Surface_Point(plane, numpy_normal, collision_point)

    else:
        last_point = surface_point[:]
        surface_point.clear()
        surface_point.append(new_p[0]-numpy_normal[0]*0.5)
        surface_point.append(new_p[1]-numpy_normal[1]*0.5)
        surface_point.append(new_p[2]-numpy_normal[2]*0.5)

        direct_vec = np.cross(numpy_normal, [0, -1, 0])
        normal_plane = direct_vec[0]*(x-new_p[0])+direct_vec[1]*(y-new_p[1])+direct_vec[2]*(z-new_p[2])

        intersect = plane
        const = -direct_vec[0]*new_p[0]-direct_vec[1]*new_p[1]-direct_vec[2]*new_p[2]
        if direct_vec[2] != 0:
            subsititude = (direct_vec[0]*x+direct_vec[1]*y+const)/direct_vec[2]
            intersect = intersect.subs(z, subsititude)
        elif direct_vec[1] != 0:
            subsititude = (direct_vec[0]*x+direct_vec[2]*z+const)/direct_vec[1]
            intersect = intersect.subs(y, subsititude)
        elif direct_vec[0] != 0:
            subsititude = (direct_vec[1]*y+direct_vec[2]*z+const)/direct_vec[0]
            intersect = intersect.subs(x, subsititude)
        #print(intersect)
        poly_x = Poly(intersect,x).all_coeffs()
        poly_y = Poly(intersect,y).all_coeffs()
        poly_z = Poly(intersect,z).all_coeffs()
        
        coeff_xx = float(poly_x[0]) if len(poly_x)==3 else 0.0
        coeff_yy = float(poly_y[0]) if len(poly_y)==3 else 0.0
        coeff_zz = float(poly_z[0]) if len(poly_z)==3 else 0.0
        coeff_x = intersect.coeff(x)
        coeff_y = intersect.coeff(y)
        coeff_z = intersect.coeff(z)
        coeff_xy = float(coeff_x.coeff(y))
        coeff_yz = float(coeff_y.coeff(z))
        coeff_zx = float(coeff_z.coeff(x))
        coeff_x = float(coeff_x) if coeff_x.is_number else float(coeff_x.func(*[term for term in coeff_x.args if not term.free_symbols]))
        coeff_y = float(coeff_y) if coeff_y.is_number else float(coeff_y.func(*[term for term in coeff_y.args if not term.free_symbols]))
        coeff_z = float(coeff_z) if coeff_z.is_number else float(coeff_z.func(*[term for term in coeff_z.args if not term.free_symbols]))
        constant = float(intersect.func(*[term for term in intersect.args if not term.free_symbols]))

        coeff_a = float(normal_plane.coeff(x))
        coeff_b = float(normal_plane.coeff(y))
        coeff_c = float(normal_plane.coeff(z))
        coeff_d = float(const)

        # Gradients of the Lagrange function
        # dfdL = grad(plane_Lagrange, 0)
        
        # sol = fsolve(plane_obj, [1.0, -0.5, 0.5, 1.0])
        
        # print(coeff_xx)
        # print(coeff_yy)
        # print(coeff_zz)
        # print(coeff_xy)
        # print(coeff_yz)
        # print(coeff_zx)
        # print(coeff_x)
        # print(coeff_y)
        # print(coeff_z)
        # print(constant)
        # print(coeff_a)
        # print(coeff_b)
        # print(coeff_c)
        # print(coeff_d)
        
        direct_vec = np.cross(direct_vec, numpy_normal)
        direct_vec = direct_vec / np.linalg.norm(direct_vec)

        new_point1 = []
        new_point2 = []

        collision_point = collision_trace[len(collision_trace)-1]
        point = surface_point[:]
        dfdL = grad(line_Lagrange, 0)
        sol = root(line_obj, [collision_point[0], collision_point[1], collision_point[2], 1.0, 1.0], method='lm')
        new_point1 = np.array(sol.x[0:3])

        # result = solve([normal_plane, plane], [y])

        # if isinstance(result, list):
        #     if surface_point[0] >= 0:
        #         x_val = result[1][0].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #         y_val = result[1][1].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #         z_val = result[1][2].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #         if x_val.is_real and y_val.is_real and z_val.is_real:
        #             new_point1.append(float(x_val))
        #             new_point1.append(float(y_val))
        #             new_point1.append(float(z_val))
        #         else:
        #             result = solve([normal_plane, plane], [z,y,x])
        #             x_val = result[1][2].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #             y_val = result[1][1].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #             z_val = result[1][0].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #             new_point1.append(float(x_val))
        #             new_point1.append(float(y_val))
        #             new_point1.append(float(z_val))

        #     else:
        #         x_val = result[0][0].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #         y_val = result[0][1].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #         z_val = result[0][2].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #         if x_val.is_real and y_val.is_real and z_val.is_real:
        #             new_point1.append(float(x_val))
        #             new_point1.append(float(y_val))
        #             new_point1.append(float(z_val))
        #         else:
        #             result = solve([normal_plane, plane], [z,y,x])
        #             x_val = result[0][2].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #             y_val = result[0][1].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #             z_val = result[0][0].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])
        #             new_point1.append(float(x_val))
        #             new_point1.append(float(y_val))
        #             new_point1.append(float(z_val))
        # else:
        #     if x in result:
        #         new_point1.append(float(result[x].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])))
        #     else:
        #         new_point1.append(float(surface_point[0]))
        #     if y in result:
        #         new_point1.append(float(result[y].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])))
        #     else:
        #         new_point1.append(float(surface_point[1]))
        #     if z in result:
        #         new_point1.append(float(result[z].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])))
        #     else:
        #         new_point1.append(float(surface_point[2]))

        surface_point2 = []
        surface_point2.append(new_point1[0]+new_v[0]*dt)
        surface_point2.append(new_point1[1]+new_v[1]*dt)
        surface_point2.append(new_point1[2]+new_v[2]*dt)

        point = surface_point2[:]
        sol = root(line_obj, [collision_point[0], collision_point[1], collision_point[2], 1.0, 1.0], method='lm')
        collision_point = []
        new_point2 = np.array(sol.x[0:3])

        # if isinstance(result, list):
        #     if surface_point2[0] >= 0:
        #         x_val = result[1][0].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #         y_val = result[1][1].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #         z_val = result[1][2].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #         if x_val.is_real and y_val.is_real and z_val.is_real:
        #             new_point2.append(float(x_val))
        #             new_point2.append(float(y_val))
        #             new_point2.append(float(z_val))
        #         else:
        #             result = solve([normal_plane, plane], [z,y,x])
        #             x_val = result[1][2].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #             y_val = result[1][1].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #             z_val = result[1][0].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #             new_point2.append(float(x_val))
        #             new_point2.append(float(y_val))
        #             new_point2.append(float(z_val))
        #     else:
        #         x_val = result[0][0].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #         y_val = result[0][1].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #         z_val = result[0][2].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #         if x_val.is_real and y_val.is_real and z_val.is_real:
        #             new_point2.append(float(x_val))
        #             new_point2.append(float(y_val))
        #             new_point2.append(float(z_val))
        #         else:
        #             result = solve([normal_plane, plane], [z,y,x])
        #             x_val = result[0][2].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #             y_val = result[0][1].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #             z_val = result[0][0].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])
        #             new_point2.append(float(x_val))
        #             new_point2.append(float(y_val))
        #             new_point2.append(float(z_val))
        # else:
        #     if x in result:
        #         new_point2.append(float(result[x].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])))
        #     else:
        #         new_point2.append(float(surface_point2[0]))
        #     if y in result:
        #         new_point2.append(float(result[y].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])))
        #     else:
        #         new_point2.append(float(surface_point2[1]))
        #     if z in result:
        #         new_point2.append(float(result[z].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])))
        #     else:
        #         new_point2.append(float(surface_point2[2]))

        # print(new_point1)
        # print(new_point2)
        new_direc = np.array([new_point2[0]-new_point1[0],new_point2[1]-new_point1[1],new_point2[2]-new_point1[2]])
        l = np.linalg.norm(new_direc)
        second_angle = 0.0
        new_direc = new_direc / np.linalg.norm(new_direc)
        if not (direct_vec[0]*new_direc[0]>=0 and direct_vec[2]*new_direc[2]>=0):
            direct_vec = direct_vec*-1
        first_angle = vector_angle(direct_vec)
        second_angle = vector_angle(new_direc)
        # print(first_angle)
        # print(second_angle)
        angle_diff = 0.0
        if direct_vec[1] * new_direc[1] < 0:
            angle_diff = first_angle + second_angle
        elif direct_vec[1] < 0:
            angle_diff = first_angle - second_angle
        else:
            angle_diff = second_angle - first_angle
        # print(angle_diff)
        new_vec = np.zeros([3])
        for i in range(3):
            new_vec[i] = new_point1[i] - last_point[i]


        curvature = angle_diff / np.linalg.norm(new_vec)

        new_surface_point = []
        if  angle_diff > 1e-6:
            collision_point.append(float(last_point[0]+numpy_normal[0]*distance))
            collision_point.append(float(last_point[1]+numpy_normal[1]*distance))
            collision_point.append(float(last_point[2]+numpy_normal[2]*distance))
            o = np.array([float(collision_point[0]+numpy_normal[0]*1/curvature),float(collision_point[1]+numpy_normal[1]*1/curvature),float(collision_point[2]+numpy_normal[2]*1/curvature)])

            collision_point.clear()
            new_normal = np.array([float(o[0]-new_p[0]), float(o[1]-new_p[1]), float(o[2]-new_p[2])])
            new_distance = np.linalg.norm(new_normal)

            new_normal = new_normal / new_distance

            surface_point.clear()
            surface_point.append(float(new_p[0]-new_normal[0]*formula.r))
            surface_point.append(float(new_p[1]-new_normal[1]*formula.r))
            surface_point.append(float(new_p[2]-new_normal[2]*formula.r))

            numpy_normal = new_normal[:]

        poly_x = Poly(plane,x).all_coeffs()
        poly_y = Poly(plane,y).all_coeffs()
        poly_z = Poly(plane,z).all_coeffs()
        
        coeff_xx = float(poly_x[0]) if len(poly_x)==3 else 0.0
        coeff_yy = float(poly_y[0]) if len(poly_y)==3 else 0.0
        coeff_zz = float(poly_z[0]) if len(poly_z)==3 else 0.0
        coeff_x = plane.coeff(x)
        coeff_y = plane.coeff(y)
        coeff_z = plane.coeff(z)
        coeff_xy = float(coeff_x.coeff(y))
        coeff_yz = float(coeff_y.coeff(z))
        coeff_zx = float(coeff_z.coeff(x))
        coeff_x = float(coeff_x) if coeff_x.is_number else float(coeff_x.func(*[term for term in coeff_x.args if not term.free_symbols]))
        coeff_y = float(coeff_y) if coeff_y.is_number else float(coeff_y.func(*[term for term in coeff_y.args if not term.free_symbols]))
        coeff_z = float(coeff_z) if coeff_z.is_number else float(coeff_z.func(*[term for term in coeff_z.args if not term.free_symbols]))
        constant = float(plane.func(*[term for term in plane.args if not term.free_symbols]))
        point = surface_point[:]
        dfdL = grad(plane_Lagrange, 0)

        collision_point = collision_trace[len(collision_trace)-1]
        # sol = fsolve(plane_obj, [collision_point[0], collision_point[1], collision_point[2], 1.0])
        sol = root(plane_obj, [collision_point[0], collision_point[1], collision_point[2], 1.0], method='lm')
        # print(sol)
        collision_point = []
        collision_point = np.array(sol.x[0:3])
        new_direction = collision_point - surface_point
        new_distance = np.linalg.norm(new_direction)
        if new_direction[0]*numpy_normal[0] >= 0 and new_direction[1]*numpy_normal[1] >= 0 and new_direction[2]*numpy_normal[2] >= 0:
            new_direction = 1
        else:
            new_direction = -1

        if new_direction == 1 and inited == 1 and (new_distance > distance):
            new_p = np.array([new_p[0]-new_v[0]*dt, new_p[1]-new_v[1]*dt, new_p[2]-new_v[2]*dt])

            temp_p = collision_point+numpy_normal*(0.5-distance)
            new_v = (temp_p - new_p)/dt

            surface_point.clear()
            surface_point.append(collision_point[0]-numpy_normal[0]*distance)
            surface_point.append(collision_point[1]-numpy_normal[1]*distance)
            surface_point.append(collision_point[2]-numpy_normal[2]*distance)

            temp_p = temp_p.astype(np.float32)
            new_v = new_v.astype(np.float32)
            new_p = temp_p[:]

        if not any(np.equal(collision_trace, collision_point).all(1)):
            collision_trace = np.append(collision_trace, np.array([collision_point]), axis=0)
            line.set_lines(collision_trace)
        if new_direction == -1:
            is_collision = 0
        else:
            direction = new_direction
            distance = new_distance


start_time = time.monotonic()
init()
theta = acos((225 + 281.25 - 56.25)/(2*15*7.5* math.sqrt(5)))
surface.pos.copy_from(surface_x)
numpy_surface = surface.pos.to_numpy()
shape = numpy_surface.shape
numpy_surface = numpy_surface.reshape((shape[0]*shape[1], shape[2]))
numpy_normal = np.cross(numpy_surface[0] - numpy_surface[1], numpy_surface[1] - numpy_surface[2])
numpy_normal = numpy_normal / np.linalg.norm(numpy_normal)
F = 0
formula = tina.PrimitiveFormulation(tmodel, 10)

while gui.running:
    scene.input(gui)
    # print("theta")
    # print(scene.control.theta)
    # print("phi")
    # print(scene.control.phi)
    # print("radius")
    # print(scene.control.radius)

    if gui.is_pressed('r'):
        init()

    tmodel.set_transform(tina.translate(p[0].value) @ rot_matrix)

    scene.render()
    gui.set_image(scene.img)
    if gui.frame % time_k == 0:
        gui.show(f'{int(gui.frame/time_k):06d}.png')
    else:
        gui.show()

    if gui.frame >= (240*time_k):
        np.set_printoptions(threshold=np.inf)
        np.savetxt('collision_trace.txt', collision_trace)
        print(timedelta(seconds=time.monotonic() - start_time))
        break

    if not gui.is_pressed(gui.SPACE):
        for i in range(steps):
            if next_colli_t > 0:
                next_colli_t -= 1
            else:
                tina_substep()
            substep(gravity, float(formula.mass), is_collision, numpy_normal, float(distance), float(direction),new_p,new_v)




