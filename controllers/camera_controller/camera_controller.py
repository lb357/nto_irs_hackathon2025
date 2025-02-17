#=====================================#
#         CAMERA CONTROLLER           #
#=====================================#
# Estimated Time: 28 sim min          #
#=====================================#
# Can be improved:                    #
# 1. Replacing GRM+BFS with PRM+A*    #
# 2. Add pathfinding for ArUco-marker #
#=====================================#
#        Team "План Б -которого нет-" #
#               Хакатон НТО ИРС, 2025 #
#=====================================#


from controller import Robot, Camera
import numpy as np
import numpy.typing as npt
from typing import Tuple
import cv2
import socket
import threading
from collections import deque
import pickle

CPREFIX = "CAM"

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

MARKER_SIZE = 0.1
MARKER_IMGK = 4
H = 2
S = 2
ROBOT_H = 0.008
MARKER_H = 0.1
ROBOTS_IDS = {9:1, 10:2}
EPSILONK = 10**(-10)#0.00001
EPSILONC = 0.008

MARKER_DST = 15
ROBOT_R = 25
ROBOT_ERR = 25
MARKER_COLLISION_DST = 35
gridRoadmapStep = 10

ROBOT2_POS = (round(0+ROBOT_R), round(640-ROBOT_R))

HOST = "127.0.0.1"
ROTO_1_PORT = 27001
ROTO_2_PORT = 27002


def estimatePoseSingleMarkers(marker_points, marker_size, cameraMatrix, distCoeffs):
    marker_world_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    return cv2.solvePnP(marker_world_points, marker_points, cameraMatrix, distCoeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)

def get3DMarkerCorners(marker_size, rvec, tvec):
    half_side = marker_size/2
    rot_mat, jacobian_mat = cv2.Rodrigues(rvec)

    marker_world_points = np.array([[-marker_size / 2, marker_size / 2, 0, 1],
                              [marker_size / 2, marker_size / 2, 0, 1],
                              [marker_size / 2, -marker_size / 2, 0, 1],
                              [-marker_size / 2, -marker_size / 2, 0, 1]], dtype=np.float32)
    mat = np.array([
        [rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], tvec[0][0]],
        [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], tvec[1][0]],
        [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], tvec[2][0]],
        [0, 0, 0, 1]
    ])
    marker_camera_points = np.array([
        np.dot(mat, marker_world_points[0]),
        np.dot(mat, marker_world_points[1]),
        np.dot(mat, marker_world_points[2]),
        np.dot(mat, marker_world_points[3])
    ])
    return marker_camera_points[:, :-1]

def get_lab_img(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return frame
    
def get_pmp(u, v, z, fx, fy, cx, cy):
    x = (u-cx)*z / fx
    y = (v-cy)*z / fy
    return x, y
    
def get_psp(x, y, z, fx, fy, cx, cy, is_rounded = False):
    u = fx * x / z + cx
    v = fy * y / z + cy
    if is_rounded:
        return round(u), round(v)
    else:
        return u, v

def get_pmf(u, v, s, w, h, is_rounded = False):
    x = u / (w/s)
    y = v / (h/s)
    x -= s/2
    y -= s/2
    if is_rounded:
        return round(x), round(y)
    else:
        return x, y
        
def get_psf(x, y, s, w, h, is_rounded = False):
    x += s/2
    y += s/2
    u = x * (w/s)
    v = y * (h/s)
    if is_rounded:
        return round(u), round(v)
    else:
        return u, v

def set_pos(con, x, y, th):
    con.sendall(pickle.dumps([0, x, y, th]))
    
def set_target(con, xt, yt):
    con.sendall(pickle.dumps([1, xt, yt]))

def uFindContours(image: npt.ArrayLike, mode: int, method: int) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    try:
        _, contours, hierarchies = cv2.findContours(image, mode, method)
        return contours, hierarchies
    except Exception as err1:
        try:
            contours, hierarchies = cv2.findContours(image, mode, method)
            return contours, hierarchies
        except Exception as err2:
            raise Exception(f"Error in cv2.findContours: {err1} (legacy {err2})")

def get_mask(hsv, min_h, min_s, min_v, max_h, max_s, max_v, del_contours = []):
    img_mask = cv2.inRange(hsv,
        np.array(hsv2int(min_h, min_s, min_v), dtype=np.uint8),
        np.array(hsv2int(max_h, max_s, max_v), dtype=np.uint8)
    )
    mask_kernel = np.ones((5, 5), dtype=np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, mask_kernel)
    if len(del_contours) > 0:
        for contour in del_contours:
            img_mask = cv2.drawContours(img_mask, contour.astype(np.int32), -1, (0, 0, 0), thickness=-1)
    mask_contours, mask_contours_hierarchy = uFindContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in mask_contours:
        img_mask = cv2.drawContours(img_mask, [contour], -1, (255, 255, 255), thickness=-1)
        contour_hull = cv2.convexHull(contour)
        epsilon = EPSILONK * cv2.arcLength(contour_hull, True)
        contour_approx = cv2.approxPolyDP(contour_hull, epsilon, True)
        img_mask = cv2.drawContours(img_mask, [contour_approx], -1, (255, 255, 255), thickness=-1)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, mask_kernel)
    return img_mask
    

def hsv2int(h, s, v):
    return h / 360 * 179, s / 100 * 255, v / 100 * 255

def int2hsv(h, s, v):
    return h / 179 * 360, s / 255 * 100, v / 255 * 100
    
def mass_center(contour: npt.ArrayLike, is_rounded: bool = True) -> Tuple[float, float]:
    moments = cv2.moments(np.array(contour, dtype=np.float32))
    if moments['m00'] != 0:
        x = float(moments['m10'] / moments['m00'])
        y = float(moments['m01'] / moments['m00'])
        if is_rounded:
            return round(x), round(y)
        else:
            return x, y
    else:
        return None
        
def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2))

def get_map_mask(mask, obstacles = [], r = 0):
    mask = cv2.bitwise_not(mask)
    mask_kernel = np.ones((3, 3), dtype=np.uint8)
    for pos in obstacles:
        mask = cv2.circle(mask, pos, round(1.5*r), (0, 0, 0), thickness=-1)
    mask = cv2.erode(mask, mask_kernel,iterations = round(0.5*r))
    return mask

def bfs(start: tuple, goal: tuple, graph: dict):
    queue = deque([start])
    visited = {start: None}
    while queue:
        cur_node = queue.popleft()
        if cur_node == goal:
            break
        next_nodes = graph[cur_node]
        for next_node in next_nodes:
            if next_node not in visited:
                queue.append(next_node)
                visited[next_node] = cur_node
    return visited


def nearest_vertex(point: tuple, graph: dict):
    dst = -1
    nearest = ()
    for vertex in graph:
        vd = ((vertex[0]-point[0])**2 + (vertex[1]-point[1])**2)**(1/2)
        if dst == -1 or vd < dst:
            dst = vd
            nearest = vertex
    return nearest


def invert_path(from_point, path):
    #print(from_point, path)
    #if from_point in path:
    if from_point in path:
        if path[from_point] is None or path[from_point] == ():
            return [from_point]
        else:
            return invert_path(path[from_point], path) + [from_point]
    else:
        return invert_path(path[from_point], path) + [from_point]
    #return [from_point]


def nbfs(start: tuple, goal: tuple, graph: dict):
    start_ = nearest_vertex(start, graph)
    goal_ = nearest_vertex(goal, graph)
    path_ = bfs(start_, goal_, graph)
    ipath_ = invert_path(goal_, path_)
    if start != start_ and goal != goal_:
        return [start] + ipath_ + [goal]
    elif start != start_:
        return [start] + ipath_
    elif goal != goal_:
        return ipath_ + [goal]
    else:
        return ipath_
        
def get_roadmap(img, w, h):
    roadmap = {}
    for y in range(0, h, gridRoadmapStep):
        for x in range(0, w, gridRoadmapStep):
            if img[y][x] != 0:
                p = (x, y)
                if p not in roadmap:
                    roadmap[p] = []
                
                if y+gridRoadmapStep < len(img):
                    if img[y+gridRoadmapStep][x]:
                        roadmap[p].append((x, y+gridRoadmapStep))
                
                if x+gridRoadmapStep < len(img[y]):
                    if img[y][x+gridRoadmapStep]:
                        roadmap[p].append((x+gridRoadmapStep, y))
                if y-gridRoadmapStep >= 0:
                    if img[y-gridRoadmapStep][x]:
                        roadmap[p].append((x, y-gridRoadmapStep))
                if x-gridRoadmapStep >= 0:
                    if img[y][x-gridRoadmapStep]:
                        roadmap[p].append((x-gridRoadmapStep, y))
    return roadmap

def get_normal_pathpoint(pos, path):
    for i in range(len(path)):
        if distance(*pos, *path[i]) >= ROBOT_ERR:
            return i
    return i 

#with open('param.txt') as f:
#    K = eval(f.readline())
#    D = eval(f.readline())

K=np.array([[640, 0.0, 320], [0.0, 640, 320], [0.0, 0.0, 1.0]])
D=np.array([[0.0], [0.0], [0.0], [0.0]])
with open('task.pkl', "rb") as f:
    out_markers = list(pickle.loads(f.read()))

print(CPREFIX, f"Out markers: {out_markers}")

parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementWinSize = 0

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


robot1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
robot1.connect((HOST, ROTO_1_PORT))
robot2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
robot2.connect((HOST, ROTO_2_PORT))
ROBOTS_CONNECTIONS = {1:robot1, 2:robot2}

robot = Robot()
timestep = int(robot.getBasicTimeStep())

cam: Camera = robot.getDevice('camera')
cam.enable(timestep)

markers = {}
robots = {}

storage_mask = np.array([])
storages = {}
inp_mask = np.array([])
out_mask = np.array([])

captured_markers = {}
is_sorted = False
target_1, target_2 = (0, 0), (0, 0)

while robot.step(timestep) != -1:
    data = cam.getImage()
    img = np.frombuffer(data, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
    img = cv2.undistort(img, K, D)
    imgs = img.copy()
    temp = get_lab_img(img)
    temp = cv2.resize(temp, (img.shape[1]*MARKER_IMGK, img.shape[0]*MARKER_IMGK), interpolation=cv2.INTER_AREA)
    temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp_corners, ids, temp_rejected = detector.detectMarkers(temp_gray)
    corners = tuple([corner//MARKER_IMGK for corner in temp_corners])
    img = cv2.resize(temp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    h, w, _ = img.shape
    
    hsv = cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)
    
    if ids is not None:
        for ix in range(len(ids)):
             points = corners[ix][0].astype(np.int16).tolist()
             idx = ids[ix][0]
             mret, rvec, tvec = estimatePoseSingleMarkers(corners[ix], MARKER_SIZE, K, D)
             if mret:
                 marker_center, jacobian_marker_center = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec, tvec,K,D)
                 wmarker_center = marker_center[0][0].astype(np.int16).tolist()
                 
                 mp = get3DMarkerCorners(MARKER_SIZE, rvec, tvec)
                 mpc = [(mp[0][0] + mp[1][0] + mp[2][0] + mp[3][0])/4, (mp[0][1] + mp[1][1] + mp[2][1] + mp[3][1])/4]
                 mps = [(mp[1][0] + mp[2][0])/2, (mp[1][1] + mp[2][1])/2]
                 dmp = (mps[0]-mpc[0], mps[1]-mpc[1])

                 angle = -np.degrees(np.arctan2(dmp[1], dmp[0]))
                       
                 #print(idx, mpc, angle)
                                  
                 smpc = get_psp(*mpc, tvec[2][0], K[0][0], K[1][1], K[0][2], K[1][2], True)
                 smps = get_psp(*mps, tvec[2][0], K[0][0], K[1][1], K[0][2], K[1][2], True)
                 
                 if idx in ROBOTS_IDS:
                     smpb = get_psp(*mpc, tvec[2][0]+ROBOT_H, K[0][0], K[1][1], K[0][2], K[1][2], True)#get_psf(*mpc, S, w, h, True)
                     robots[ROBOTS_IDS[idx]] = [*mpc, angle, smpb, tvec[2][0]+ROBOT_H]
                     set_pos(ROBOTS_CONNECTIONS[ROBOTS_IDS[idx]], *mpc, angle)
                 else:
                     smpb = get_psp(*mpc, tvec[2][0]+MARKER_H, K[0][0], K[1][1], K[0][2], K[1][2], True)
                     rmpc = (round(mpc[0], 3), round(mpc[1], 3))
                     markers[idx] = [rmpc, angle, smpb, tvec[2][0]+MARKER_H]
                     
                 img = cv2.circle(img, smpb, 5, (0, 255, 0), cv2.FILLED)          
                 img = cv2.circle(img, smpc, 5, (255, 0, 0), cv2.FILLED)
                 img = cv2.circle(img, smps, 5, (0, 0, 255), cv2.FILLED)
                 
    if len(storage_mask) == 0:
        storage_mask = get_mask(hsv, 0, 0, 75, 360, 10, 95, corners)
        storage_mask_ = storage_mask.copy()
        storage_mask_on_start = storage_mask.copy()
        storage_contours, storage_contours_hierarchy = uFindContours(storage_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in storage_contours:
            storages[mass_center(contour, True)] = contour
          
    if len(inp_mask) == 0:
        inp_mask = get_mask(hsv, 150, 55, 50, 280, 100, 100, corners)  
        inp_mask_ = inp_mask.copy()
    if len(out_mask) == 0:
        out_mask = get_mask(hsv, 30, 55, 50, 100, 100, 100, corners)  
        out_mask_ = out_mask.copy()
        out_contours, out_contours_hierarchy = uFindContours(out_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
        out_pos = mass_center(out_contours[0]) 
    
    storage_mask = storage_mask_.copy()
    inp_mask = inp_mask_.copy()
    out_mask = out_mask_.copy()
    
    stored_markers = []
    unstored_markers = []
    full_storages = []
    free_storages = list(storages.keys())
    for marker in markers:
        marker_ipos = markers[marker][2]
        is_stored = False
        for storage in free_storages.copy():
            if distance(*marker_ipos, *storage) <= MARKER_DST:
                free_storages.remove(storage)
                storage_mask = cv2.drawContours(storage_mask, [storages[storage]], -1, (0, 0, 0), thickness=-1)
                is_stored = True
                full_storages.append(storage)
                break
        if is_stored:
            stored_markers.append(marker)
        else:
            unstored_markers.append(marker)
    
    unout_markers = []
    for marker in out_markers:
        if marker in markers:
            marker_ipos = markers[marker][2]
            if distance(*marker_ipos, *out_pos) > 3*MARKER_DST:
                unout_markers.append(marker)

    free_storages = list(set(free_storages) - set(full_storages))
    full_storage_mask = cv2.bitwise_and(storage_mask_on_start, cv2.bitwise_not(storage_mask))
    #obstacles_mask = cv2.bitwise_or(cv2.bitwise_or(full_storage_mask, inp_mask), out_mask)
    #obstacles_mask = cv2.bitwise_or(full_storage_mask, out_mask)
    obstacles_mask = full_storage_mask
    for marker in markers:
        marker_ipos = markers[marker][2]
        if marker in captured_markers:
            obstacles_mask = cv2.circle(obstacles_mask, marker_ipos, MARKER_COLLISION_DST, (255, 255, 255), cv2.FILLED)
        else:
            obstacles_mask = cv2.circle(obstacles_mask, marker_ipos, round(1.5*MARKER_COLLISION_DST), (255, 255, 255), cv2.FILLED)
    #print(free_storages)
    
    map_mask_1 = get_map_mask(obstacles_mask, [robots[2][3]], ROBOT_R)
    map_mask_2 = get_map_mask(obstacles_mask, [robots[1][3]], ROBOT_R)
    roadmap_1 = get_roadmap(map_mask_1, w, h)
    roadmap_2 = get_roadmap(map_mask_2, w, h)
    #debug_img = np.full(shape=(640, 640), fill_value = 255, dtype=np.uint8)
    #for pos in roadmap_2:
    #    debug_img = cv2.circle(debug_img, pos, 1, 0, -1)
    #print(robots[2][3], (0, h))
    
    if len(captured_markers) == 0:
        if len(unstored_markers) == 0:
            is_sorted = True
            
            
        if not is_sorted:
            if len(unstored_markers) > 0:
                captured_markers[unstored_markers[0]] = free_storages[0]
        else:
            if len(unout_markers) > 0:
                captured_markers[unout_markers[0]] = out_pos
            elif len(unstored_markers) > 0:
                captured_markers[unstored_markers[0]] = free_storages[0]
    for marker in captured_markers.copy():
        if marker in stored_markers and not is_sorted:
            del captured_markers[marker]
        elif marker not in unout_markers and is_sorted:
            del captured_markers[marker]
        elif marker in stored_markers and marker not in out_markers and is_sorted:
            del captured_markers[marker]
        else:
            mpos = markers[marker][2]
            spos = captured_markers[marker]
            dx, dy = spos[0]-mpos[0], -(spos[1]-mpos[1])
            tangle = np.arctan2(dy, dx)
            cx, cy = np.cos(tangle+np.pi)*MARKER_COLLISION_DST, np.sin(tangle+np.pi)*MARKER_COLLISION_DST
            tpoint = (round(mpos[0]+cx), round(mpos[1]-cy))
            if distance(*robots[1][3], *tpoint) >= ROBOT_ERR:
                try:
                    img = cv2.circle(img, tpoint, 6, (0, 255, 255), cv2.FILLED)
                    #print(mpos, captured_markers[marker],  dx, dy, np.degrees(tangle), tpoint)
                    path_1 = np.array(nbfs(robots[1][3], tpoint, roadmap_1), dtype=np.int32)
                    path_1_epsilon = EPSILONC * cv2.arcLength(path_1, True)
                    path_1 = cv2.approxPolyDP(path_1, path_1_epsilon, False)
                    path_1 = np.reshape(path_1, (len(path_1), 2))
                    nppi = get_normal_pathpoint(robots[1][3], path_1)
                    targetm_1 = path_1[nppi]
                    target_1 = get_pmp(*targetm_1, robots[1][4], K[0][0], K[1][1], K[0][2], K[1][2])
                    
                    img = cv2.line(img, robots[1][3], targetm_1, (0, 255, 0), 2)
                    img = cv2.circle(img, targetm_1, 5, (255, 255, 0), cv2.FILLED)
                    for dpoint in range(nppi, len(path_1)):
                        img = cv2.line(img, path_1[dpoint-1], path_1[dpoint], (255, 255, 0), 1)
                    img = cv2.circle(img, path_1[-1], 5, (255, 255, 0), cv2.FILLED)
                except Exception as err:
                    target_1 = get_pmp(*tpoint, robots[1][4], K[0][0], K[1][1], K[0][2], K[1][2])
            else:
                target_1 = get_pmp(*tpoint, robots[1][4], K[0][0], K[1][1], K[0][2], K[1][2])
    
    path_2 = np.array(nbfs(robots[2][3], ROBOT2_POS, roadmap_2), dtype=np.int32)
    path_2_epsilon = EPSILONC * cv2.arcLength(path_2, True)
    path_2 = cv2.approxPolyDP(path_2, path_2_epsilon, False)
    path_2 = np.reshape(path_2, (len(path_2), 2))
    targetm_2 = path_2[get_normal_pathpoint(robots[2][3], path_2)]
    print(CPREFIX, f"Unstored markers: {unstored_markers}, Sorted: {is_sorted}, Unout_markers: {unout_markers}")
    #img = cv2.line(img, robots[2][3], targetm_2, (255, 255, 0), 1)
    #img = cv2.circle(img, targetm_2, 5, (255, 255, 0), cv2.FILLED)
    #for dpoint in range(2, len(path_2)):
    #    img = cv2.line(img, path_2[dpoint-1], path_2[dpoint], (255, 255, 0), 1)
    #img = cv2.circle(img, path_2[-1], 5, (255, 255, 0), cv2.FILLED)
    target_2 = get_pmp(*targetm_2, robots[2][4], K[0][0], K[1][1], K[0][2], K[1][2])
        
        
        
   
    #print(get_roadmap(map_mask_1, w, h))
    
    #print(CPREFIX, f"\nRobots: {robots}\nMarkers: {markers}\n\n")
    #r1p = get_psp(-0.25, 0.25, 2, K[0][0], K[1][1], K[0][2], K[1][2], True)
    #r2p = get_psp(*target_2, 2, K[0][0], K[1][1], K[0][2], K[1][2], True)
    #img = cv2.circle(img, r1p, 5, (255, 255, 0), cv2.FILLED)
    set_target(ROBOTS_CONNECTIONS[1], *target_1)
    set_target(ROBOTS_CONNECTIONS[2], *target_2)

            
    cv2.imshow("DISPLAY", img)
    #cv2.imshow("DEBUG", debug_img)
    #cv2.imshow("DEBUG", obstacles_mask)#img)
    cv2.waitKey(1)