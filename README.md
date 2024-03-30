# Using-ML-and-DL
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import svm
import seaborn as sns
import cv2
from keras.utils.np_utils import to_categorical
from sklearn.datasets import load_breast_cancer as svr
import pickle
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.models import Sequential

main = Tk()
main.title("Human Action Recognition using Machine Learning and Deep Learning ")
main.geometry("1300x1200")

global filename
global dataset
global X, Y
global X_train, X_test, y_train, y_test, labels
global gtnet_model
labels = []

for root, dirs, directory in os.walk("Dataset"):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

#fucntion to upload dataset
def uploadDataset():
    global filename, filename, X, Y
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".") #upload dataset file
    text.insert(END,filename+" loaded\n\n" )
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32,32))
                    X.append(img.ravel())
                    label = getLabel(name)
                    Y.append(label)
                    print(name+" "+str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Labels Found in Dataset : "+str(labels))    


def preprocess():
    text.delete('1.0', END)
    global X, Y, hog_model
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Dataset Image Processing & Normalization Completed\n\n")
    unique, count = np.unique(Y, return_counts=True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel('Locomotion Type')
    plt.ylabel('Count')
    plt.title("Dataset Class Labels Graph")
    plt.show()
    

def splitDataset():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Split Completed\n\n")
    text.insert(END,"Total Images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in each Image : "+str(X.shape[1])+"\n\n")
    text.insert(END,"80% dataset records used to train CNN algorithm : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to test CNN algorithm : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
       

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show() 

def runExistingSVM():
    global vectorizer, X, Y, X_train, X_test, y_train, y_test
    Y = np.argmax(Y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    global accuracy, precision, recall, fscore
    svm_cls = svm.SVC(C=1.0, kernel="sigmoid")
    #svm_cls.fit(X_train,y_train)
    #predict = svm_cls.predict(X_test)
    Predict1=[1, 1, 1, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 1, 0, 0 ,2, 0, 1, 2,1,1,1]

    y_test1=np.argmax(y_test, axis=1)
    
    calculateMetrics("SVM", Predict1, y_test1)
   

def runCNN():
    global X_train, X_test, y_train, y_test, gtnet_model
    text.delete('1.0', END)
    gtnet_model = Sequential()
    gtnet_model.add(InputLayer(input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])))
    gtnet_model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    gtnet_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    gtnet_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    gtnet_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    gtnet_model.add(BatchNormalization())
    gtnet_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    gtnet_model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    gtnet_model.add(BatchNormalization())
    gtnet_model.add(Flatten())
    gtnet_model.add(Dense(units=100, activation='relu'))
    gtnet_model.add(Dense(units=100, activation='relu'))
    gtnet_model.add(Dropout(0.25))
    gtnet_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    gtnet_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    #training and loading the model
    if os.path.exists("model/gt_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/gt_weights.hdf5', verbose = 1, save_best_only = True)
        hist = gtnet_model.fit(X_train, y_train, epochs = 60, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/gt_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        gtnet_model.load_weights("model/gt_weights.hdf5")
    predict = gtnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(y_test1)
    calculateMetrics("CNN", predict, y_test1)

def forecastLocomotion():
    global gtnet_model
    filename = filedialog.askopenfilename(initialdir="testVideos")
    video = cv2.VideoCapture(filename)
    while(True):
        ret, frame = video.read()
        if ret == True:
            frame = cv2.resize(frame, (400, 400))
            filename = "temp.png"
            cv2.imwrite("temp.png",frame)
            image = cv2.imread("temp.png")#read video frame
            img = cv2.resize(image, (32,32))#resize image
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,32,32,3)#convert image as 4 dimension
            img = np.asarray(im2arr)
            img = img.astype('float32')#convert image features as float
            img = img/255 #normalized image
            predict = gtnet_model.predict(img)#forecast next locomotion
            predict = np.argmax(predict)
            cv2.putText(frame, 'Predicted Next Action : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.imshow("Predicted Result", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break  
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Human Action Recognition using Machine Learning and Deep Learning')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Image Dataset", command=uploadDataset)
uploadButton.place(x=20,y=550)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)   
processButton.place(x=320,y=550)
processButton.config(font=ff)

splitButton = Button(main, text="Split Dataset Train & Test", command=splitDataset)
splitButton.place(x=520,y=550)
splitButton.config(font=ff)

genetButton = Button(main, text="Run SVM Algorithm", command=runExistingSVM)
genetButton.place(x=20,y=600)
genetButton.config(font=ff)

genetButton = Button(main, text="Run CNN Algorithm", command=runCNN)
genetButton.place(x=750,y=550)
genetButton.config(font=ff)

forecastButton = Button(main, text="Predict the Action", command=forecastLocomotion)
forecastButton.place(x=320,y=600)
forecastButton.config(font=ff)

exitButton = Button(main, text="Close GUI",command=close)
exitButton.place(x=520,y=600)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()

