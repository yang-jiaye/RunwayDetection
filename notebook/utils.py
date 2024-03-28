import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Input: 
    2 coordinates of line segment
Output: 
    Hough parameters theta and rho, and length of line
    theta is the slope angle of normal line of line
    rho is the distance to origin with sign depending on sign of x intersect 
    range of theta is within -pi/2 to pi/2
    range of rho is within -maxRho to maxRho, where maxRho is the length of diagnoal line of image
    To distinguish 2 line segments with same distance to origin and slope but in different sides of origin,
    we define that for line with negative x intersect, rho is negative with abs equal to distance to origin. 
'''
def computeHoughTransformCoord(x1, y1, x2, y2):

    if y1 == y2:
        theta = np.pi/2
        rho = abs(y1)
        length = abs(x1 - x2)
        return theta, rho, length

    k_inv =  (x1 - x2) / (y1 - y2)
    theta = np.arctan(-k_inv)

    A = y2 - y1
    B = x2 - x1
    C = abs(x2*y1 - x1*y2)
    length = np.sqrt(A**2 + B**2)
    rho = C / length

    # distinguish same distance and theta but different sides
    X0 = - k_inv * y1 + x1
    if X0 < 0:
        rho = -rho
    
    return theta, rho, length

'''
compute average gray scale around a line
'''
def average_grayscale_along_line(image, x1, y1, x2, y2):
    # Calculate the length of the line
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Calculate the unit direction vector
    vx = (x2 - x1) / L
    vy = (y2 - y1) / L
    
    # Initialize a list to store average grayscale values
    grayscale_values = []
    
    # Iterate over each point along the line
    for i in range(int(L)):
        x = int(x1 + i * vx)
        y = int(y1 + i * vy)
        
        # Iterate over 5 points to the left and 5 points to the right of the current point
        for j in range(-9, 10):
            x_neighbor = x + j
            if x_neighbor >= 0 and x_neighbor < image.shape[1]\
                and y >= 0 and y < image.shape[0]:
                grayscale_value = image[y, x_neighbor]
                grayscale_values.append(grayscale_value)
        
    average_grayscale = np.mean(grayscale_values)
    
    return average_grayscale


'''
Input:
    grid: a list contains (theta, rho, line, d) tuple
    color: the color of runway
    draw: whether to draw result
Output:
    (theta, rho): Hough parameters
'''
def computeResult(grid: list, lineImg, color = (0, 0, 255), draw=False):
    # 假设 max_index 是一个包含多个元组 (theta, rho) 的列表
    thetas = [pair[0] for pair in grid]
    rhos = [pair[1] for pair in grid]
    lines = [pair[2] for pair in grid]
    ds = [pair[3] for pair in grid]

    # thetas = np.asarray(thetas)
    # rhos = np.asarray(rhos)
    # ds = np.asarray(ds)

    # theta = np.sum(thetas * ds)/np.sum(ds)
    # rho = np.sum(rhos * ds)/np.sum(ds)

    # m = 1/np.tan(-theta)
    # c = rho / np.sin(theta)

    x = []
    y = []
    weights = []
    for line, d in zip(lines, ds):
        line = line[0]
        x.append(line[0])
        x.append(line[2])
        y.append(line[1])
        y.append(line[3])
        weights.append(d)
        weights.append(d)

    A = np.vstack([x, np.ones(len(x))]).T
    w = np.diag(np.sqrt(weights))
    m, c = np.linalg.lstsq(w @ A, y @ w, rcond=None)[0]

    if(m == 0):
        theta = np.pi/2
        rho = c
    else:
        theta = np.arctan(-1/m)
        rho = c * np.sin(theta)

    if draw == True:
        # draw result lines
        if theta == 0:
            cv2.line(lineImg, (rho, -1000), (rho, 1000), color, 1)
        elif not np.isnan(theta):
            LL = 3000
            cv2.line(lineImg, (0, int(c)), (LL, int( LL *m + c)), color, 1)

        # draw grid lines    
        for line, d in zip(lines, ds):
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(lineImg, (x1, y1), (x2, y2), (255, 0, 255), 1)

    return theta, rho

'''
Input: 
    Image to detect
Output:
    line0, line1: 2 runwayline parameters containing (theta, rho)
'''
def detectAirportRunway1(img, draw=False, res_name="draw_result"):
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # gray = hsv[:,:,0]
    if img is None:
        print("[RunwayDetect]: Image empty")
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect line segmentation
    lsd = cv2.createLineSegmentDetector(scale=0.8, sigma_scale=0.6, quant=2.0, ang_th=22.5, log_eps=0.0, density_th=0.7, n_bins=1024)
    lines, _, prec, nfa = lsd.detect(gray)

    # set threshold
    length_thres = 2
    gray_thres = 50

    thetaList = []
    dList = []
    rhoList = []
    lineList = []
    for line in lines:
        # check average gray scale
        x1, y1, x2, y2 = map(int, line[0])
        average_grayscale = average_grayscale_along_line(gray, x1, y1, x2, y2)
        if average_grayscale < gray_thres:
            continue

        theta, rho, d = computeHoughTransformCoord(x1, y1, x2, y2)
        if d < length_thres:
            continue
        if abs(np.abs(theta) - np.pi/2) < np.pi / 16 or np.abs(theta) > 1.5707963:
            continue
        thetaList.append(theta)
        dList.append(d)
        rhoList.append(rho)
        lineList.append(line)

    # set grid numbers, which is related to size of image
    # to do: find relation between grid num and image size
    if width * height < 100000:
        N = 40
    else:
        N = 100
    Ntheta = N
    Nrho = N
    maxRho = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    # 创建一个网格
    theta_bins = np.linspace(-np.pi/2, np.pi/2, num=Ntheta)
    rho_bins = np.linspace(-maxRho, maxRho, num=Nrho)
    totalLengthGrid = np.zeros((len(theta_bins) - 1, len(rho_bins) - 1))
    thetaThoGrid = [[[] for _ in range(len(rho_bins) - 1)] for _ in range(len(theta_bins) - 1)]

    # 映射thetaList和rhoList到网格中
    for theta, rho, d, line in zip(thetaList, rhoList, dList, lineList):
        theta_idx = np.digitize(theta, theta_bins) - 1
        rho_idx = np.digitize(rho, rho_bins) - 1
        totalLengthGrid[theta_idx, rho_idx] += d
        thetaThoGrid[theta_idx][rho_idx].append((theta, rho, line, d))

    # 找到最大元素的索引
    max_index = np.unravel_index(np.argmax(totalLengthGrid), totalLengthGrid.shape)
    totalLengthGrid[max_index] = -np.inf

    # 找到第二大元素的索引
    second_max_index = np.unravel_index(np.argmax(totalLengthGrid), totalLengthGrid.shape)
    totalLengthGrid[second_max_index] = -np.inf

    # 找到第三大元素的索引
    third_max_index = np.unravel_index(np.argmax(totalLengthGrid), totalLengthGrid.shape)

    lineImg = img.copy()
    line0 = computeResult(thetaThoGrid[max_index[0]][max_index[1]], lineImg, color = (0, 0, 255), draw = draw)
    line1 = computeResult(thetaThoGrid[second_max_index[0]][second_max_index[1]], lineImg, color = (0, 255, 0), draw = draw)

    if draw == True:
        cv2.imwrite("images_result/"+res_name+".png", lineImg)
    # cv2.imwrite("test.png", lineImg)

    return line0, line1

def detectAirportRunway2(img, draw=False, res_name="draw_result", rho_step=3, theta_step=np.pi/180 * 3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, _ = img.shape
    
    lsd = cv2.createLineSegmentDetector(scale=0.8, sigma_scale=0.6, quant=2.0, ang_th=22.5, log_eps=0.0, density_th=0.7, n_bins=1024)
    lines, _, prec, nfa = lsd.detect(gray)

    length_thres = 2
    gray_thres = 0

    lineList = []
    for line in lines:
        # check average gray scale
        x1, y1, x2, y2 = map(int, line[0])
        average_grayscale = average_grayscale_along_line(gray, x1, y1, x2, y2)
        if average_grayscale < gray_thres:
            continue
        if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < length_thres:
            continue
        if (x1 != x2) and ( abs((y2 - y1)/(x2 - x1)) < np.tan(np.pi/8) ):
            continue

        lineList.append(line)

    mask = np.zeros([height, width], dtype=np.uint8)
    for line in lineList:
        x1, y1, x2, y2 = map(int, line[0])
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
    
    # use opencv hough transform
    lines = cv2.HoughLines(mask, rho_step, theta_step, threshold=10)

    # number of lines to select
    line_num = 2

    if lines is None:
        return None, None
    
    sorted_lines = sorted(lines[:line_num], key=lambda x: x[0][1], reverse=True)
    line_L = sorted_lines[0][0]
    line_R = sorted_lines[1][0]

    if lines is not None:
        for line in lines[:line_num]:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), 255, 1)
    if draw == True:
        cv2.imwrite("images_result/"+res_name+".png", img)
    
    return line_L, line_R

if __name__ == "__main__":
    img = cv2.imread("images/3.png")
    detectAirportRunway2(img, draw=True)