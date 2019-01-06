import cv2 as cv
import numpy as np
from PIL import Image


def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result



f=open('train/ISIC-2017_Training_Part3_GroundTruth.csv', 'r')
f.readline()

i=1
for line in f.readlines():
    print(i)
    i+=1
    file=line.split(",")[0]
    melanoma=line.split(",")[1]
    keratosis=line.split(",")[2]
    img = cv.imread("train/ISIC-2017_Training_Data/"+ file +".jpg")
    img = cv.resize(img, (500, 500))
    final = white_balance(img)
        
    if float(melanoma)==1:
        cv.imwrite("train/normalizedTrain/trainMelanoma/true/"+file+".jpg", final)

        
        #normImage = Image.open("train/normalizedTrain/trainMelanoma/true/"+file+".jpg")

        #normImage = normImage.rotate(90)
        #normImage.save("train/normalizedTrain/trainMelanoma/true/"+file+"_90.jpg")

        #normImage = normImage.rotate(90)
        #normImage.save("train/normalizedTrain/trainMelanoma/true/"+file+"_180.jpg")

        #normImage = normImage.rotate(90)
        #normImage.save("train/normalizedTrain/trainMelanoma/true/"+file+"_360.jpg")

        #normImage.close()
        

    else:
        cv.imwrite("train/normalizedTrain/trainMelanoma/false/"+file+".jpg", final)

    if float(keratosis)==1:
        cv.imwrite("train/normalizedTrain/trainKeratosis/true/"+file+".jpg", final)

        
        #normImage = Image.open("train/normalizedTrain/trainKeratosis/true/"+file+".jpg")

        #normImage = normImage.rotate(90)
        #normImage.save("train/normalizedTrain/trainKeratosis/true/"+file+"_90.jpg")

        #normImage = normImage.rotate(90)
        #normImage.save("train/normalizedTrain/trainKeratosis/true/"+file+"_180.jpg")

        #normImage = normImage.rotate(90)
        #normImage.save("train/normalizedTrain/trainKeratosis/true/"+file+"_360.jpg")

        #normImage.close()


    else:
        cv.imwrite("train/normalizedTrain/trainKeratosis/false/"+file+".jpg", final)

f.close()



f=open('validation/ISIC-2017_Validation_Part3_GroundTruth.csv', 'r')
f.readline()

i=1
for line in f.readlines():
    print(i)
    i+=1
    file=line.split(",")[0]
    melanoma=line.split(",")[1]
    keratosis=line.split(",")[2]
    img = cv.imread("validation/ISIC-2017_Validation_Data/"+ file +".jpg")
    img = cv.resize(img, (500, 500))
    final = white_balance(img)

        
    if float(melanoma)==1:
        cv.imwrite("validation/normalizedValidation/validationMelanoma/true/"+file+".jpg", final)
    else:
        cv.imwrite("validation/normalizedValidation/validationMelanoma/false/"+file+".jpg", final)

    if float(keratosis)==1:
        cv.imwrite("validation/normalizedValidation/validationKeratosis/true/"+file+".jpg", final)
    else:
        cv.imwrite("validation/normalizedValidation/validationKeratosis/false/"+file+".jpg", final)

f.close()





f=open('test/ISIC-2017_Test_v2_Part3_GroundTruth.csv', 'r')
f.readline()

i=1
for line in f.readlines():
    print(i)
    i+=1
    file=line.split(",")[0]
    melanoma=line.split(",")[1]
    keratosis=line.split(",")[2]
    img = cv.imread("test/ISIC-2017_Test_v2_Data/"+ file +".jpg")
    img = cv.resize(img, (500, 500))
    final = white_balance(img)

        
    if float(melanoma)==1:
        cv.imwrite("test/normalizedTest/testMelanoma/true/"+file+".jpg", final)
    else:
        cv.imwrite("test/normalizedTest/testMelanoma/false/"+file+".jpg", final)

    if float(keratosis)==1:
        cv.imwrite("test/normalizedTest/testKeratosis/true/"+file+".jpg", final)
    else:
        cv.imwrite("test/normalizedTest/testKeratosis/false/"+file+".jpg", final)

f.close()
