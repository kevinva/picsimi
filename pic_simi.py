import cv2
import os
import sys
import getopt
import time
import numpy as np
import matplotlib.pyplot as plt

CLUSTER_COUNT = 10   ## 可调参

SCAN_ALL = False  # 主要扫描全部图片，聚类数目最好改改

KEY_WORDS = ['arrow', 'back', 'close', 'add', 'indicator']

TEST_CASES = ['ic_room_pick_heart_guest_pop_close',
              'hotline_exit_close_icon_pressed',
              'chat_send_gift_dialog_close',
              'me_ic_nav_close',
              'me_ic_tag_close',
              'room_emotion_back',
              'me_ic_nav_back',
              'icon_new_nav_back',
              'me_ic_nav_back_white',
              'me_glamour_backgroundIMG',
              'onekey_loginBack']

def isAllBlackImage(image):
    isBlack = True
    row, col = image.shape
    for r in range(row):
        for c in range(col):
            if image[r, c] != 0:
                isBlack = False
                break
    return isBlack

def checkMayUseAlpha(image):
    row, col = image.shape
    useIt = False
    for r in range(row):
        for c in range(col):
            if image[r, c] != 0:
                useIt = True
                break
    return useIt

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


def getImagesPathsWith(dirPath):
    imagePathList = list()
    for item in os.listdir(dirPath):
        itemPath = dirPath + '/' + item
        if os.path.isfile(itemPath):
            if itemPath.endswith('2x.png'):  # 忽略3x图片，用2x即可
                imagePathList.append(itemPath)

        else:
            subPathList = getImagesPathsWith(itemPath)
            if len(subPathList) > 0:
                imagePathList.extend(subPathList)


    return imagePathList


def generateStatisticsPage(resultInfo):
    filePathWritten = './result_{}.html'.format(int(time.time()))
    with open(filePathWritten, 'w') as file:
        lines = list()
        lines.append('<html>')
        lines.append('  <body style="background-color:rgba(204, 102, 0, 0.25)">')
        for category, paths in resultInfo.items():
            lines.append('      <h3>{}</h3>'.format(category))
            lines.append('      <table>')
            for path in paths:
                pathSegments = path.split('/')
                assert len(pathSegments) > 0
                imageName = pathSegments[-2].replace('.imageset', '')
                lines.append('          <tr>')
                lines.append("              <td><img src='{}' /></td>".format(path))
                lines.append("              <td>{}</td>".format(imageName))
                lines.append('          </tr>')
            lines.append('      </table>')
            lines.append('      <hr>')
        lines.append('  </body>')
        lines.append('</html>')

        file.writelines(lines)


def generateImageDataAt(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print('image is None')
        return None
    
    a0, a1, a2 = image.shape
    if a2 == 4: # 有些图片可能没alpha通道
        bChannel, gChannel, rChannel, alpahChannel = cv2.split(image)
    # print('alpha: ', alpahChannel)

    ## 先做高斯模糊，使图片像素分布更加平滑，增加计算容忍度
    imageBlur = cv2.GaussianBlur(image, (3, 3), 1)
    imageGray = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
    if a2 == 4 and checkMayUseAlpha(alpahChannel):
        alpha = 0.7  # 可调参
        imageGray = np.int16((1 - alpha) * imageGray + alpha * alpahChannel)
    # print('imageGray: ', imageGray)

    ## 有的图可能灰度之后全是黑色（如c9_8.png），可考虑其alpha通道（png图有RGBA四通道）
    ## 有时不得不考虑下alpha通道（如hotline_exit_close_icon_pressed@2x.png，其alpha通道才正确反映轮廓特征，反而RGB不能）
    if a2 == 4 and isAllBlackImage(imageGray):
        # print('path', path)
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

    # Test
    # imageTest = cv2.resize(imageDxy, (18, 18), interpolation=cv2.INTER_CUBIC) 
    # print('file: ', path)
    # print(imageTest)
    ## Test

    imageNew = np.zeros(imageDxy.shape, dtype=np.int16)
    for i, rowList in enumerate(imageDxy):
        for j, col in enumerate(rowList):
            if imageDxy[i][j] > 50:
                imageNew[i][j] = 1
    # print('imageNew: ',imageNew)

    ## 去掉上下左右空白部分
    ## 可调参
    topBound, leftBound, bottomBound, rightBound = findImageEdge(imageNew)
    imageNew = imageNew[topBound: bottomBound + 1, leftBound: rightBound + 1]
    # print('imageNew2: ', imageNew2)

    imageNew = cv2.resize(imageNew, (38, 38), interpolation=cv2.INTER_CUBIC) ### # 可调参！目前用（38, 38）准确度最高！！！

    imageOut = imageNew.reshape(-1)
    imageOut = np.float32(imageOut)
    # print(imageOut)

    # imageOut = cv2.cvtColor(imageDxy, cv2.COLOR_GRAY2RGB)
    # plt.imshow(imageOut)
    # plt.show()

    return imageOut


def app(argv):
    inputPath = ''
    try:
        opts, args = getopt.getopt(argv, 'hi:', ['ifile='])
    except getopt.GetoptError:
        print("try 'python main.py -i <xxx.xcassets folder path>'")
        sys.exit(2)

    if len(opts) == 0:
        print("try 'python main.py -i <xxx.xcassets folder path>'")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("try 'python main.py -i <xxx.xcassets folder path>'")
            sys.exit()
        elif opt in ('-i', '--ifile'):
            inputPath = arg

    if len(inputPath) == 0:
        print('Input path is none!')
        sys.exit()


    imagePathList = getImagesPathsWith(inputPath)
    imagePathListClean = list()
    imageNameList = list()
    imageList = list()
    for filePath in imagePathList:
        pathSegments = filePath.split('/')
        assert len(pathSegments) > 0
        imageName = pathSegments[-2].replace('.imageset', '')
        
        if SCAN_ALL:
            imageNameList.append(imageName)
            imagePathListClean.append(filePath)
        else:
            # 过滤一下
            for keyWord in KEY_WORDS:
                if keyWord in imageName.lower():
                    imageNameList.append(imageName)
                    imagePathListClean.append(filePath)

    print('Total images: {}, names: {}'.format(len(imagePathListClean), len(imageNameList)))

    # 再过滤一下
    if SCAN_ALL:
        imagePathListClean2 = imagePathListClean
        imageNameList2 = imageNameList
    else:
        imagePathListClean2 = list()
        imageNameList2 = list()
        for filePath, imageName in zip(imagePathListClean, imageNameList):  
            imageCheck = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
            a0, a1, a2 = imageCheck.shape
            if a0 > 120 or a1 > 120:  # 暂不考虑较大尺寸的图
                continue

            imagePathListClean2.append(filePath)
            imageNameList2.append(imageName)
    print('Total images2: {}, names2: {}'.format(len(imagePathListClean2), len(imageNameList2)))

    for filePath in imagePathListClean2:
        imageData = generateImageDataAt(filePath)
        if imageData is not None:
            imageList.append(imageData)

    # imageList = list()
    # imageNameList = list()
    # for filename in os.listdir('./data1/'):
    #     filePath = './data1/' + filename
    #     imageData = generateImageDataAt(filePath)
    #     if imageData is not None:
    #         imageList.append(imageData)
    #         imageNameList.append(filename)

    imageArr = np.array(imageList)
    # print(imageArr.shape)

    critereia = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(data=imageArr,
                                      K=CLUSTER_COUNT,
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
        groupList.append(imagePathListClean2[index])
    
    # for key, value in result.items():
    #     print('key: ', key)
    #     print(value)
    
    generateStatisticsPage(result)
    print('Done!')


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
    app(sys.argv[1:])
    # test3()
    # generateImageDataAt('./data2/hotline_exit_close_icon_pressed@2x.png')
