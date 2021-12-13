import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def isAllBlackImage(image):
    isBlack = True
    row, col = image.shape
    for r in range(row):
        for c in range(col):
            if image[r, c] != 0:
                isBlack = False
                break
    return isBlack

def findImageEdge(image):
    row, col = image.shape
    topBound = 0
    bottomBound = row - 1
    leftBound = 0
    rightBound = col - 1
    topFound = False
    bottomFound = False
    leftFound = False
    rightFound = False
    for r in range(row):
        for c in range(col):
            if image[r, c] != 0:
                topBound = r
                topFound = True
                break
        if topFound:
            break

    for r in reversed(range(row)):
        for c in range(col):
            if image[r, c] != 0:
                bottomBound = r
                bottomFound = True
                break
        if bottomFound:
            break
        
    for c in range(col):
        for r in range(row):
            if image[r, c] != 0:
                leftBound = c
                leftFound = True
                break
        if leftFound:
            break

    for c in reversed(range(col)):
        for r in range(row):
            if image[r, c] != 0:
                rightBound = c
                rightFound = True
                break 
        if rightFound:
            break
    return topBound, leftBound, bottomBound, rightBound


def generateImageDataAt(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print('image is None')
        return None
    
    # image = cv2.resize(image, (18, 18), interpolation=cv2.INTER_CUBIC)
    bChannel, gChannel, rChannel, alpahChannel = cv2.split(image)
    # print('alpha: ', alpahChannel)

    ## 先做高斯模糊，使图片像素分布更加平滑，增加计算容忍度
    imageBlur = cv2.GaussianBlur(image, (3, 3), 1)
    imageGray = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
    # print('imageGray: ', imageGray)

    ## 有的图可能灰度之后全是黑色（如c9_8.png），可考虑其alpha通道（png图有RGBA四通道）
    if isAllBlackImage(imageGray):
        print('path', path)
        imageGray = alpahChannel
        imageGray = cv2.GaussianBlur(imageGray, (3, 3), 1)
        # print('use alpha channel: ', imageGray)

    ## 用梯度图，过滤掉一些有统一色值的背景（如c9_1.png）
    sobelx = cv2.Sobel(imageGray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(imageGray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    imageDxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # print('imageDxy: ', imageDxy)

    imageNew = np.zeros(imageDxy.shape, dtype=np.int16)
    for i, rowList in enumerate(imageDxy):
        for j, col in enumerate(rowList):
            if imageDxy[i][j] > 50:
                imageNew[i][j] = 1
    # print('imageNew: ',imageNew)

    ## 去掉上下左右空白部分
    topBound, leftBound, bottomBound, rightBound = findImageEdge(imageNew)
    imageNew = imageNew[topBound: bottomBound + 1, leftBound: rightBound + 1]
    # print('imageNew2: ', imageNew2)
    imageNew = cv2.resize(imageNew, (38, 38), interpolation=cv2.INTER_CUBIC) ### 目前用（38, 38）准确度最高！！！

    imageOut = imageNew.reshape(-1)
    imageOut = np.float32(imageOut)
    # print(imageOut.shape)

    # imageOut = cv2.cvtColor(imageDxy, cv2.COLOR_GRAY2RGB)
    # plt.imshow(imageOut)
    # plt.show()

    return imageOut


def app():
    imageList = list()
    imageNameList = list()
    for filename in os.listdir('./data1/'):
        filePath = './data1/' + filename
        imageData = generateImageDataAt(filePath)
        if imageData is not None:
            imageList.append(imageData)
            imageNameList.append(filename)

    imageArr = np.array(imageList)
    # print(imageArr.shape)

    critereia = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, (centers) = cv2.kmeans(data=imageArr,
                                      K=k,
                                      bestLabels=None,
                                      criteria=critereia,
                                      attempts=20,
                                      flags=cv2.KMEANS_RANDOM_CENTERS
                                      )
    # print('centers: ', centers.shape)
    # print('label: ', labels.flatten().shape)

    result = dict()
    for index, l in enumerate(labels.flatten()):
        k = str(l)
        groupList = result.get(k)
        if groupList is None:
            groupList = list()
            result[k] = groupList
        groupList.append(imageNameList[index])
    
    for key, value in result.items():
        print('key: ', key)
        print(value)
        
def test3():
    image = cv2.imread('./data1/c9_2.png', cv2.IMREAD_UNCHANGED)
    bChannel, gChannel, rChannel, alpahChannel = cv2.split(image)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(imageGray, (18, 18), interpolation=cv2.INTER_CUBIC)
    # print(image)
    imageBlur = cv2.GaussianBlur(image, (3, 3), 1)
    # print(imageBlur)

    sobelx = cv2.Sobel(imageBlur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(imageBlur, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    imageDxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    print('imageDxy: \n', imageDxy)

    # imageOut = cv2.cvtColor(imageGray, cv2.COLOR_GRAY2RGB)
    # plt.imsave('./temp/3.png', imageOut)

    return

if __name__ == '__main__':
    # app()
    test3()
    # generateImageDataAt('./data1/c10_1.png')
