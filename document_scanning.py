import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import scipy.special as special

"""載入圖片"""
img1 = cv2.imread("0001.jpg", -1)

# cv2.imshow("Original Image", img1)

"""做Canny邊緣偵測"""
nr, nc = img1.shape[:2]
img2 = img1.copy()
img3 = np.zeros([nr, nc])
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)

# cv2.imshow("Canny Edge Detection", edges)

"""做霍夫直線轉換找邊界"""
lines = cv2.HoughLines(edges, 1, math.pi/180.0, 325)
# 0001:325 0002:300
print(lines.shape[0])
if lines is not None:
    a, b, c = lines.shape
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0, y0 = a*rho, b*rho
        pt1 = (int(x0+2000*(-b)), int(y0+2000*a))
        pt2 = (int(x0-2000*(-b)), int(y0-2000*a))
        cv2.line(img3, pt1, pt2, 255, 1, cv2.LINE_AA)
hough_line_detection = img3.copy()

# cv2.imshow("Hough Line Detection", hough_line_detection)

"""利用邊界交點找四個頂點"""
pt = [[], [], [], [], []]
quadrant1_pt = []
quadrant2_pt = []
quadrant3_pt = []
quadrant4_pt = []

for i in range(1, nr-1):
    for j in range(1, nc-1):
        # 上下左右皆為白色(邊界交點)
        if img3[i + 1][j] == 255 and img3[i-1][j] == 255 \
                and img3[i][j+1] == 255 and img3[i][j-1] == 255:
            # 找與原始圖片各角的距離
            dis = [(i - nr) ** 2 + (j - nc) ** 2, i ** 2 + j ** 2,
                   i ** 2 + (j - nc) ** 2, (i - nr) ** 2 + j ** 2]
            # 找與最靠近的角的距離
            dis = min(dis)
            if i < (nr/2) and j < (nc/2):        # 位於左上半部
                pt[1].append([j, i, dis])
            elif i < (nr/2) and j > (nc/2):      # 位於右上半部
                pt[2].append([j, i, dis])
            elif i > (nr / 2) and j < (nc / 2):  # 位於左下半部
                pt[3].append([j, i, dis])
            elif i > (nr / 2) and j > (nc / 2):  # 位於右下半部
                pt[4].append([j, i, dis])
            # cv2.circle(img3, (j, i), 10, 255, 10)
print(pt)
# 獲取列表的第三個元素
def takeThird(elem):
    return elem[2]

# 以與靠近角距離最短的點作為實際切割角點，四個分區各取一點
pt_final = []
for i in range(1, 5):
    if len(pt[i]) > 1:
        pt[i].sort(key=takeThird)  # 指定以第三个元素排序
        pt_final.append(pt[i][0])
        cv2.circle(img3, (pt[i][0][0], pt[i][0][1]), 10, 255)
    else:
        pt_final.append(pt[i][0])
        cv2.circle(img3, (pt[i][0][0], pt[i][0][1]), 10, 255)
print(pt_final)

# cv2.imshow("Get Cornor", img3)

"""做仿射轉換"""
pts1 = np.float32([pt_final[0][:2], pt_final[1][:2], pt_final[2][:2], pt_final[3][:2]])
pts2 = np.float32([[0, 0], [1050, 0], [0, 1485], [1050, 1485]])  # A4尺寸 210mm×297mm => 1050:1485
T = cv2.getPerspectiveTransform(pts1, pts2)
img4 = cv2.warpPerspective(img1, T, (1050, 1485))

# cv2.imshow( "Perspective Transform", img4)

"""修正光源不均"""
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5 = img4.copy()
blur = 9
for i in range(0, 1485-blur, blur):
    for j in range(0, 1050-blur, blur):
        values = []
        for k1 in range(blur):
            for k2 in range(blur):
                values.append(img4[i+k1][j+k2])
        maximum = max(values)  # 找到blur內最亮的點(白紙的部分)
        err = 255-maximum-10  #
        for k1 in range(blur):
            for k2 in range(blur):
                img5[i+k1][j+k2] += err

"""將字以外的部分都變成白色(255)"""
for i in range(1485):
    for j in range(1050):
        if img5[i][j] >= 225:
            img5[i][j] = 255

# cv2.imshow("mysolution", img5)


"""做Gamma校正"""
def gamma_correction(f, gamma = 2.0):
    g = f.copy()
    nr, nc = f.shape[:2]
    c = 255.0 / (255.0 ** gamma)
    table = np.zeros(256)
    for i in range(256):
        table[i] = round(i ** gamma * c, 0)
    if f.ndim != 3:
        for x in range(nr):
            for y in range(nc):
                g[x, y] = table[f[x, y]]
    else:
        for x in range(nr):
            for y in range(nc):
                for k in range(3):
                    g[x, y, k] = table[f[x, y, k]]
    return g

img6 = gamma_correction(img5, 2.0)

# cv2.imshow("gamma_correction", img6)


"""印圖"""
cv2.imshow("Original Image", img1)
cv2.imshow("Canny Edge Detection", edges)
cv2.imshow("Hough Line Detection", hough_line_detection)
cv2.imshow("Get Cornor", img3)
cv2.imshow("Perspective Transform", img4)
cv2.imshow("Correction", img5)
cv2.imshow("gamma_correction(final)", img6)


cv2.imwrite("Original Image.jpg", img1)
cv2.imwrite("Canny Edge Detection.jpg", edges)
cv2.imwrite("Hough Line Detection.jpg", hough_line_detection)
cv2.imwrite("Get Cornor.jpg", img3)
cv2.imwrite("Perspective Transform.jpg", img4)
cv2.imwrite("Correction.jpg", img5)
cv2.imwrite("gamma_correction(final).jpg", img6)

cv2.waitKey(0)