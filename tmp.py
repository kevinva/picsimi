def test():
    image = cv2.imread('./data/temp/IMG_8020.JPG')
    print('image: ', image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    # axes.axis('off')
    # axes.imshow(image)
    # plt.show()
    pixelValues = image.reshape((-1, 3))
    pixelValues = np.float32(pixelValues)
    print('pixelValues: ', pixelValues.shape)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, (centers) = cv2.kmeans(data =pixelValues, 
                                    K = k,
                                    bestLabels = None,
                                    criteria = criteria,
                                    attempts = 10,
                                    flags = cv2.KMEANS_RANDOM_CENTERS
                                    )
    centers = np.uint8(centers)
    print('centers: ', centers.shape)
    print('labels unflatten:', labels)
    labels = labels.flatten()
    print('labels flatten: ', labels)
    segmented_image = centers[labels]
    print('segmented: ', segmented_image)
    segmented_image = segmented_image.reshape(image.shape)
    # plt.imshow(segmented_image)
    # plt.show()

    masked_image = np.copy(image)
    masked_image = masked_image.reshape((-1, 3))
    masked_image[labels == 0] = [255, 255, 2555]
    masked_image = masked_image.reshape(image.shape)
    plt.imshow(masked_image)
    plt.show()

def correntPercentage(result):
    pass

# 直接用图片像素特征，准确度不行
def test2():
    imageNameList = list()
    imageList = list()
    for index, filename in enumerate(os.listdir('./data')):
        filePath = './data/' + filename
        image = cv2.imread(filePath)
        
        if image is not None:
            image = cv2.resize(image, (40, 40), interpolation=cv2.INTER_CUBIC)
            image = image.reshape(-1, 3)
            imageNameList.append(filename)
            imageList.append(np.float32(image))

    # imageTest = cv2.cvtColor(imageDict['./data/c2_1.png'], cv2.COLOR_BGR2RGB)
    # plt.imshow(imageTest)
    # plt.show()
    # print(imageTest.shape)
    # print(imageDict.values().shape)


    imageArray = np.array(imageList)
    print('imageArray: ', imageArray.shape)

    critereia = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, (centers) = cv2.kmeans(data=imageArray,
                                      K=k,
                                      bestLabels=None,
                                      criteria=critereia,
                                      attempts=10,
                                      flags=cv2.KMEANS_RANDOM_CENTERS
                                      )
    print('centers: ', centers.shape)
    print('label: ', labels.flatten().shape)

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