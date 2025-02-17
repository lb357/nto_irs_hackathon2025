from controller import Robot, Motor, Keyboard
import socket
import threading
import pickle
from math import *

CPREFIX = "ROBOT2"
HOST = "127.0.0.1"
PORT = 27002


R = 0.015
L = 0.075*2

KD = 75
KR = 50
KMA = 0.2
KC = 3

DP = 0.005

MIN_V = 0.5
MAX_V = 20

x, y, th = 0, 0, 0
tx, ty = 0, 0

def handler_callback(conn: socket.socket):
    global x
    global y
    global th
    global tx
    global ty
    while True:
        msg = conn.recv(1024)
        info = pickle.loads(msg)
        if info[0] == 0:
            x = info[1]
            y = info[2]
            th = radians(info[3])
        elif info[0] == 1:
            tx, ty = info[1], info[2]

        

#def get_wheel_w(r, l, v, w):
#    wl = (1/r)*(v - (w*l/2))
#    wr = (1/r)*(v - (w*l/2))
#    return wl, wr
    
def get_pos_delta(xc, yc, xt, yt, th):
    p = sqrt((xt-xc)**2 + (yt-yc)**2)
    a = atan2(yt-yc, xt-xc) + th
    #a = pi-a
    if a > pi:
        a = a - 2*pi
    if a < -pi:
        a = a + 2*pi
    return p, a
    
def get_wheel_lyapunov(p, a, k1, k2):
    v = k1*p*cos(a)
    w = k1*cos(a)*sin(a)+k2*a
    return v, w
    
def get_wheel_linear(p, a, k1, k2):
    cr = -k1*p + k2*a
    cl = -k1*p - k2*a
    return cr, cl
    
def get_wheel_const(p, a, k1, k2, k3):
    if abs(a) >= k1:
        return a*k2, -a*k2
    else:
        return -p*k3, -p*k3

    
def clamp_v(v, min_v, max_v, kc=2):
    if v < -min_v/kc:
        return max(-max_v, min(v, -min_v))
    elif v > min_v/kc:
        return max(min_v, min(v, max_v))
    else:
        return 0


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen()

conn, addr = sock.accept()
thr = threading.Thread(target=handler_callback, args=(conn,))
thr.start()

robot = Robot()
leftMotor = robot.getDevice('LeftMotor')
rightMotor = robot.getDevice('RightMotor')

leftMotor.setVelocity(0)
rightMotor.setVelocity(0)
leftMotor.setPosition(float('+inf'))
rightMotor.setPosition(float('+inf'))

timestep = int(robot.getBasicTimeStep())

while robot.step(timestep) != -1:
    p, a = get_pos_delta(x, y, tx, ty, th)
    if abs(p) >= DP:
        wl, wr = get_wheel_const(p, a, KMA, KR, KD)
        wl, wr = clamp_v(wl, MIN_V, MAX_V, KC), clamp_v(wr, MIN_V, MAX_V, KC)
        print(CPREFIX, f"wl: {wl} / wr: {wr}")
        leftMotor.setVelocity(wl)
        rightMotor.setVelocity(wr)
    else:
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
    print(CPREFIX, f"Pos:{x:.3f}/{y:.3f}, Angle:{th:.2f}, Target:{tx:.3f}/{ty:3f} | D:{p:.3f}, A:{a:.2f}")# wl, wr, p, a, th)

    #leftMotor.setVelocity(wl)
   # rightMotor.setVelocity(wr)
