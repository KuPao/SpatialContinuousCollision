from platform import dist
from sympy.core import symbol
from sympy.ntheory.residue_ntheory import _discrete_log_trial_mul
import taichi as ti
import tina
import math
import numpy as np
from sympy import *
import time
from datetime import timedelta

ti.init(ti.gpu)

gravity = 9.8
frac = 0.1
time_k = 10
dt = 5e-4 / time_k
steps = 30
angle = 90 / 180 * math.pi
rot_matrix = tina.eularXYZ([0, 0, 0])
N = 2
NN = N, N
W = 30
L = W / N
x, y, z, t = symbols('x,y,z,t')
plane = -1 * x + 2 * y
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

scene = tina.Scene(smoothing=True, taa=False, ibl=True)

roughness = tina.Param(float, initial=0)#.15)
metallic = tina.Param(float, initial=0)#.25)
material = tina.PBR(metallic=metallic, roughness=roughness)

gui = ti.GUI('matball')
roughness.make_slider(gui, 'roughness')
metallic.make_slider(gui, 'metallic')

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

@ti.kernel
def init():
    for i in ti.grouped(surface_x):
        m, n = (i + 0.5) * L - 0.5 - W/2
        surface_x[i] = ti.Vector([m, 0.5*m, n])
        # print(surface_x[i])
    for i in ti.grouped(p):
        p[i] = ti.Vector([7 - 0.5 / math.sqrt(5), 3.5 + 1 / math.sqrt(5), 0.0])
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
            acc[i] += 1600 * mass * dir * dis * ti.Vector([normal[0], normal[1], normal[2]])
        acc[i] /= mass
        v[i] += acc[i] * dt
        p[i] += dt * v[i]

def vector_angle(vec):
    ground = np.array([vec[0], 0, vec[2]])
    return acos(np.inner(vec,ground) / np.linalg.norm(vec)/np.linalg.norm(ground))

def normal_plane_curve(plane, normal_plane, v):
    if abs(v[0]) > abs(v[1]) >= abs(v[2]):
        return solve([normal_plane, plane], [z,y,x])
    elif abs(v[0]) > abs(v[2]) >= abs(v[1]):
        return solve([normal_plane, plane], [y,z,x])
    elif abs(v[1]) > abs(v[0]) >= abs(v[2]):
        return solve([normal_plane, plane], [z,x,y])
    elif abs(v[1]) > abs(v[2]) >= abs(v[0]):
        return solve([normal_plane, plane], [x,z,y])
    elif abs(v[2]) > abs(v[0]) >= abs(v[1]):
        return solve([normal_plane, plane], [y,x,z])
    else:
        return solve([normal_plane, plane], [x,y,z])
def curvature_collision_point(o, first, second, vec):
    x_v = vec[0]
    z_v = vec[2]
    a = x_v + z_v
    b = 2*(x_v*(second[0]-o[0])+z_v*(second[2]-o[2]))
    c = -((o[0]-first[0])**2+(o[1]-first[1])**2+(o[2]-first[2])**2-(o[1]-second[1])**2-o[0]**2-o[2]**2-second[0]**2-second[2]**2+2*o[0]*second[0]+2*o[1]*second[1])
    t1 = (-b + sqrt(b*b-4*a*c))/2*a
    t2 = (-b - sqrt(b*b-4*a*c))/2*a
    if abs(t1) < abs(t2):
        return np.array([second[0]+vec[0]*t1, second[1], second[2]+vec[2]*t1])
    else:
        return np.array([second[0]+vec[0]*t2, second[1], second[2]+vec[2]*t2])

def tina_substep():
    global is_collision
    global numpy_normal, surface_point, direction, distance
    global collision_trace, next_colli_p, next_colli_t
    global p, v, x, y, z, t, plane, dt
    global gui
    global new_p, new_v, inited, fra
    collision_point = []
    # return
    formula = tina.PrimitiveFormulation(tmodel, 10)
    if inited == 1:
        new_p = p.to_numpy()[0]
        new_v = np.array([v[0][0], v[0][1], v[0][2]])
    else:
        new_p = np.array([7 - 0.5 / math.sqrt(5), 3.5 + 1 / math.sqrt(5), 0.0]) + p.to_numpy()[0]
        new_v = np.array([0,0,0]) + v.to_numpy()[0]
        inited = 1
        return
    if is_collision == 0:
        collision = formula.Calculate_Collision(plane)
        for j in range(3):
            sum = 0
            for a in collision:
                midpoint = a[j].subs(x, formula.midpoint[0]).subs(y, formula.midpoint[1]).subs(z, formula.midpoint[2])
                if not midpoint.is_real:
                    F = float(formula.mass * gravity)
                    is_collision = 0
                    last_point = collision_trace[collision_trace.shape[0]-1]
                    x_val = last_point[0] + new_v[0]*t - x
                    y_val = last_point[1] + new_v[1]*t - 4.9*t*t - y
                    z_val = last_point[2] + new_v[2]*t - z
                    result = solve([plane,x_val,y_val,z_val], [t,x,y,z])
                    if not result[0][0].is_real:
                        return
                    if result[0][0] > result[1][0]:
                        next_colli_t = int(result[0][0]/dt + 1)
                    else:
                        next_colli_t = int(result[1][0]/dt + 1)
                    print(next_colli_t)
                    collision_point.clear()
                    collision_point.append(last_point[0] + new_v[0]*next_colli_t*dt)
                    collision_point.append(last_point[1] + new_v[1]*next_colli_t*dt - 4.9*next_colli_t*dt*next_colli_t*dt)
                    collision_point.append(last_point[2] + new_v[2]*next_colli_t*dt)
                    next_colli_p = np.append(next_colli_p, np.array([collision_point]), axis=0)
                    predict_line.set_lines(next_colli_p)
                    return
                sum += midpoint
            sum /= len(collision)
            collision_point.append(float(sum))
        is_collision = 1
        if not any(np.equal(collision_trace, collision_point).all(1)):
            collision_trace = np.append(collision_trace, np.array([collision_point]), axis=0)
            line.set_lines(collision_trace)
        surface_point, direction, distance = formula.Surface_Point(plane, numpy_normal, collision_point)
    
    else:
        last_point = surface_point[:]
        surface_point.clear()
        surface_point.append(last_point[0]+v[0][0]*dt)
        surface_point.append(last_point[1]+v[0][1]*dt)
        surface_point.append(last_point[2]+v[0][2]*dt)

        direct_vec = np.cross(numpy_normal, [0, -1, 0])
        normal_plane = direct_vec[0]*(x-new_p[0])+direct_vec[1]*(y-new_p[1])+direct_vec[2]*(z-new_p[2])
        direct_vec = np.cross(direct_vec, numpy_normal)
        direct_vec = direct_vec / np.linalg.norm(direct_vec)
        first_angle = vector_angle(direct_vec)


        new_point1 = []
        new_point2 = []
        result = normal_plane_curve(plane, normal_plane, new_v)
        if 'x' in result:
            new_point1.append(float(result['x'].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])))
        else:
            new_point1.append(float(surface_point[0]))
        if 'y' in result:
            new_point1.append(float(result['y'].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])))
        else:
            new_point1.append(float(surface_point[1]))
        if 'z' in result:
            new_point1.append(float(result['z'].subs(x, surface_point[0]).subs(y, surface_point[1]).subs(z, surface_point[2])))
        else:
            new_point1.append(float(surface_point[2]))
        surface_point2 = []
        surface_point2.append(new_point1[0]+direct_vec[0]*dt/1000)
        surface_point2.append(new_point1[1]+direct_vec[1]*dt/1000)
        surface_point2.append(new_point1[2]+direct_vec[2]*dt/1000)
        if 'x' in result:
            new_point2.append(float(result['x'].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])))
        else:
            new_point2.append(float(surface_point2[0]))
        if 'y' in result:
            new_point2.append(float(result['y'].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])))
        else:
            new_point2.append(float(surface_point2[1]))
        if 'z' in result:
            new_point2.append(float(result['z'].subs(x, surface_point2[0]).subs(y, surface_point2[1]).subs(z, surface_point2[2])))
        else:
            new_point2.append(float(surface_point2[2]))
        new_direc = np.array([new_point2[0]-new_point1[0],new_point2[1]-new_point1[1],new_point2[2]-new_point1[2]])
        new_direc = new_direc / np.linalg.norm(new_direc)
        second_angle = vector_angle(new_direc)
        angle_diff = second_angle - first_angle
        new_vec = np.zeros([3])
        for i in range(3):
            new_vec[i] = new_point1[i] - last_point[i]

        curvature = angle_diff / np.linalg.norm(new_vec)

        new_surface_point = []
        if abs(angle_diff) > 0.00001:
            o = []
            o.append(last_point[0]+numpy_normal[0]*1/curvature)
            o.append(last_point[1]+numpy_normal[1]*1/curvature)
            o.append(last_point[2]+numpy_normal[2]*1/curvature)
            new_point = curvature_collision_point(o, last_point, surface_point, new_v)
            new_normal = np.array([o[0]-new_point[0], o[1]-new_point[1], o[2]-new_point[2]])
            new_normal = new_normal / np.linalg.norm(new_normal)
            surface_point.clear()
            surface_point.append(float(new_point[0]-new_normal[0]*formula.r))
            surface_point.append(float(new_point[1]-new_normal[1]*formula.r))
            surface_point.append(float(new_point[2]-new_normal[2]*formula.r))
            if fra == -1:
                print(fra)
                print(first_angle)
                print(second_angle)
                print(surface_point)
                print(new_normal)
            numpy_normal = new_normal[:]

        collision_point, new_direction, new_distance = formula.Plane_Point(plane, numpy_normal, surface_point)

        # if surface_point[0] <= -8:
        #     new_direction = -1
        if new_direction == 1 and inited == 1 and (new_distance > distance):
            new_p = np.array([new_p[0]-new_v[0]*dt,new_p[1]-new_v[1]*dt,new_p[2]-new_v[2]*dt])

            temp_p = collision_point+numpy_normal*(0.5 - distance)
            new_v = (temp_p - new_p)/dt
            temp_p = temp_p.astype(np.float32)
            new_v = new_v.astype(np.float32)
            new_p = temp_p[:]
            new_distance = distance


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
# normal.from_numpy(numpy_normal)
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




