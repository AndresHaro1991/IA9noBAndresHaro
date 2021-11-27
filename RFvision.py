def a_nuevorostro(nuevorostro):
    import cv2
    import os
    import imutils

    personName = nuevorostro
    dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
    personPath = dataPath + '/' + personName
    numCamara = 0
    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)
    try:
        cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)  # 0, 1 son los índices de la cámara
		#cap = cv2.VideoCapture('Video.mp4')
    except Exception as error:
        print('Error con algo de la cámara ' + str(error))

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame =  imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
            count = count + 1
        cv2.imshow('frame',frame)

        k =  cv2.waitKey(1)
        if k == 27 or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()
    

    import cv2
    import os
    import imutils

    personName = nuevorostro
    dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
    personPath = dataPath + '/' + personName
    numCamara = 0
    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)
    try:
        cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)  # 0, 1 son los índices de la cámara
		#cap = cv2.VideoCapture('Video.mp4')
    except Exception as error:
        print('Error con algo de la cámara ' + str(error))

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame =  imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
            count = count + 1
        cv2.imshow('frame',frame)

        k =  cv2.waitKey(1)
        if k == 27 or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()

def actualizacionfacial3():
    import cv2
    import os
    import numpy as np

    dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes...')

        for fileName in os.listdir(personPath):
            print('Caras: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
		# Ver lo que se esta aprendiendo:
            image = cv2.imread(personPath+'/'+fileName,0)
            cv2.imshow('image',image)
		# eso fue lo que aprendio de
            cv2.waitKey(3)
        label = label + 1

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write('modeloLBPHFace.xml')
    print("¡Modelo almacenado!")

def actualizacionfacial2():
    import cv2
    import os
    import numpy as np

    dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes...')

        for fileName in os.listdir(personPath):
            print('Caras: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            # Ver lo que se esta aprendiendo:
            image = cv2.imread(personPath+'/'+fileName,0)
            cv2.imshow('image',image)
            # eso fue lo que aprendio de
            cv2.waitKey(3)
        label = label + 1

    #print('labels= ',labels)
    #print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
    #print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

    # Métodos para entrenar el reconocedor
    #1. face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer = cv2.face.FisherFaceRecognizer_create()
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))
    # Almacenando el modelo obtenido
    #face_recognizer.write('modeloEigenFace.xml')
    face_recognizer.write('modeloFisherFace.xml')
    #face_recognizer.write('modeloLBPHFace.xml')
    print("¡Modelo almacenado!")

def actualizacionfacial1():
    import cv2
    import os
    import numpy as np

    dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes...')

        for fileName in os.listdir(personPath):
            print('Caras: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            # Ver lo que se esta aprendiendo:
            image = cv2.imread(personPath+'/'+fileName,0)
            cv2.imshow('image',image)
            # eso fue lo que aprendio de
            cv2.waitKey(3)
        label = label + 1

    #print('labels= ',labels)
    #print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
    #print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

    # Métodos para entrenar el reconocedor
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    #2. face_recognizer = cv2.face.FisherFaceRecognizer_create()
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))
    # Almacenando el modelo obtenido
    face_recognizer.write('modeloEigenFace.xml')
    #face_recognizer.write('modeloFisherFace.xml')
    #face_recognizer.write('modeloLBPHFace.xml')
    print("¡Modelo almacenado!")

def monitorfacial():
    import cv2
    import os
    
    dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
    numCamara = 0
    imagePaths = os.listdir(dataPath)
    print('imagePaths=',imagePaths)


    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.read('modeloLBPHFace.xml')

    cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('Video.mp4')

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    while True:
        ret,frame = cap.read()
        if ret == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)
        i=0
        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            i=i+1
            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            cv2.putText(frame, 'face num '+str(i),(x+15,y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 0), 2)
		
		# LBPHFace
            if result[1] < 70:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                
            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

opcion=""
import os
os.system("cls")

while opcion!=0:
    print("Bienvenidos a Vision")
    print("Escoga la opcion que desea Ejecutar")
    print("*_____________________________________*")
    print("1.- Ingresar nuevo rostro")
    print("2.- Actualizar rostros en Sistema")
    print("3.- Reconocimiento facial")
    print("0.- Salir")
    print("*_____________________________________*")
    seleccion = int(input("Ingrese su seleccion:  "))

    if seleccion == 1:
        nuevorostro =(input("Digite el nombre de la persona:  "))
        a_nuevorostro(nuevorostro)    
        os.system("cls")
    
    elif seleccion == 2:
        print("Tipos de entrenamiento")
        print("*_____________________________________*")
        print("1.- EigenFaceRecognizer")
        print("2.- FisherFaceRecognizer")
        print("3.- LBPHFaceRecognizer")
        print("*_____________________________________*")
        tipo = int(input("Digite el tipo de entrenamiento: "))
        if tipo == 1:
            actualizacionfacial1()
            os.system("cls")
        elif tipo == 2:
            actualizacionfacial2()
            os.system("cls")
        elif tipo == 3:
            actualizacionfacial3()
            os.system("cls")
    
        
    elif seleccion == 3:
        print("Ingresando a monitor de reconocimiento facial")
#"nuevorostro="Hola"
        monitorfacial()
        os.system("cls")
    elif seleccion == 0:
        print("*_____________________________________*")
        print("*     Gracias por utilizar Vision     *")
        print("*_____________________________________*")
        break
    else:    
        print("*_____________________________________*")
        print("*             Opcion Invalida         *")
        print("*_____________________________________*")
        os.system("cls")
    
   

