import cv2
import external
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Checking OMR",layout="centered",page_icon=":memo:")

#page title 
st.title('OpenCV App for checking :red[OMR] :pencil:')

#add a button to upload the images
correct_file = st.file_uploader("Choose the correct OMR...",type=['jpg','png','jpeg'])

test_file= st.file_uploader("Choose the exam OMR...",type=['jpg','png','jpeg'])


if correct_file and test_file is not None :
    correct_file_byte = np.asarray(bytearray(correct_file.read()), dtype="uint8") 
    test_file_byte = np.asarray(bytearray(test_file.read()), dtype="uint8") 
    correct = cv2.imdecode(correct_file_byte, cv2.IMREAD_COLOR)
    test = cv2.imdecode(test_file_byte, cv2.IMREAD_COLOR)
    if st.button('show image'):

        col1, col2 = st.columns(2)
        
        with col1:
            correct_omr = st.text_input('Correct OMR')
            st.image(correct)
        with col2:
            test_omr = st.text_input('Test OMR')
            st.image(test)

    if st.button('Check omr'):
        test1 = test.copy()
        lower = np.array([60,60,60])
        higher =np.array ([250,250,250])
        mask1 = cv2.inRange(correct,lower,higher)
        mask2 = cv2.inRange(test,lower,higher)
        # st.image(mask1)

        # find contours for both the images

        cont1,_ = cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cont2,_ = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        count_img_correct=cv2.drawContours(correct,cont1,-1,255,3)
        count_img_test=cv2.drawContours(test,cont2,-1,(0,0,255),3)

        c1=max(cont1,key=cv2.contourArea)
        c2=max(cont2,key=cv2.contourArea)

        #bounding the maximum area
        x1,y1,w1,h1 = cv2.boundingRect(c1)
        cv2.rectangle(correct,(x1,y1),(x1+w1,y1+h1),(0,255,0),5)

        x2,y2,w2,h2 = cv2.boundingRect(c2)
        cv2.rectangle(test,(x2,y2),(x2+w2,y2+h2),(0,255,0),5)

        # height,width
        crop_correct= correct[y1:y1+h1,x1:x1+w1]
        crop_test= test[y2:y2+h2,x2:x2+w2]

        # Processing of images
        # compare two images and find the wrong fill bubble 
        width = 400
        height = 700
        crop_correct = cv2.resize(crop_correct,(width,height))
        crop_test = cv2.resize(crop_test,(width,height))
        # st.image(crop_correct)

        rows = 25 #question
        column = 4 #option
        correctpixcelVal = np.zeros((rows,column))
        testpixcelVal = np.zeros((rows,column))
        countCC,countTC = 0,0
        countCR,countTR = 0,0

        # APPLY THRESHOLD

        #For correct answer splitting 
        correct_gray = cv2.cvtColor(crop_correct,cv2.COLOR_BGR2GRAY)
        correct_thresh = cv2.threshold(correct_gray,0,255,cv2.THRESH_BINARY_INV)[1]

        correctBoxes = external.splitBoxes(correct_thresh)

        for Cimage in correctBoxes:
            totalCorrectPixcel = cv2.countNonZero(Cimage)
            correctpixcelVal[countCR][countCC]=totalCorrectPixcel
            countCC+=1
            if (countCC == column):
                    countCR+=1
                    countCC=0
        correctIndex = []
        for x in range(0,25):
            arr1 = correctpixcelVal[x]
            correctIndexVal = np.where(arr1==np.amax(arr1))
                
            correctIndex.append(correctIndexVal[0][0])

        # For test answer splitting

        test_gray = cv2.cvtColor(crop_test,cv2.COLOR_BGR2GRAY)
        test_thresh = cv2.threshold(test_gray,0,255,cv2.THRESH_BINARY_INV)[1]
        testBoxes = external.splitBoxes(test_thresh)

        for Timage in testBoxes:
            totalTestPixcel = cv2.countNonZero(Timage)
            testpixcelVal[countTR][countTC]=totalTestPixcel
            countTC+=1
            if (countTC == column):
                    countTR+=1
                    countTC=0
        
        testIndex = []
        for y in range(0,25):
            arr2 = testpixcelVal[y]
            testIndexVal = np.where(arr2==np.amax(arr2))
                
            testIndex.append(testIndexVal[0][0])
        #score
        display_score = ''   
        score =[]
        for x in range(0,25):
            if correctIndex[x] == testIndex[x] :
                score.append(1)
            else:
                score.append(0)
            # print(grading)
        totalScore =str("Your score is " )+str((sum(score)/25)*100)
        display_score = totalScore
        
        # print('score',totalScore)

        #display name

        markTest = test1.copy()
        testCanny = cv2.Canny(test1,10,50)
        cont2,_ = cv2.findContours(testCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(test1,cont2,-1,(0,0,255),5)
        omrRectTest= external.rectContour(cont2)
        nameRectTest = external.getCornerPoints(omrRectTest[1])
        cv2.drawContours(markTest,nameRectTest,-1,(0,0,255),10)


        nameRectTest = external.reorder(nameRectTest)
        pt01 = np.float32(nameRectTest)
        pt02= np.float32([[0,0],[325,0],[0,150],[325,150]])
        matrixN = cv2.getPerspectiveTransform(pt01,pt02)
        imgNmae = cv2.warpPerspective(test1,matrixN,(325,150))


        #displaying answer 

        imgResult = external.showAnswers(crop_test,testIndex,score,rows,column)
        imgResult = cv2.resize(imgResult,(200,600))


        col1, col2 = st.columns(2)
        with col1:
            st.success(display_score)
            st.image(imgNmae)
        with col2:
            
            st.image(imgResult)
        
        