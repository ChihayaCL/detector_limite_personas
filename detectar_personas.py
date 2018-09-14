import cv2
import numpy as np

#FRAME_WIDTH = 320
#FRAME_HEIGHT = 240
areaTH = 1000
x1 = 0
y1 = 350
x2 = 1920
y2 = y1


#Se prende la camara
cam = cv2.VideoCapture(1)
#cam.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)

#detecta sombras e ignora el resto
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True) 

#Kernels para quitar las sombras
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)


while (True):
    ret, frame = cam.read()
    
    #quitando fondo con sombras
    mascara = fgbg.apply(frame)

    #binarizando para tener un fiel contorno sin sombras
    ret,imBin= cv2.threshold(mascara,200,255,cv2.THRESH_BINARY)
    #Opening (erode->dilate) para quitar ruido.
    mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
    #Closing (dilate -> erode) para juntar regiones blancas.
    mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)

    #se dibuja la linea alertadora
    frame2 = frame
    linea1 = np.array([[x1,y1],[x2,y2]], np.int32).reshape((-1,1,2))
    frame2 = cv2.polylines(frame2,[linea1],False,(0,0,255),thickness=4)

    #contorno en las partes blancas
    _, contours0, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours0:
        
        #Dibujando rectangulos en el objeto identificado
        area = cv2.contourArea(cnt)
        print (area)
        if area > areaTH:        
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)
            circle = cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)            
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            
            #Detectado que el objeto atraviese el limite
            if (cy > 350):
                font = cv2.FONT_HERSHEY_SIMPLEX  
                color = (255,255,255)
                grosor = 2
                alerta = 'Objeto atravesando'
                
                #Escribir en pantalla el objeo atravesado
                cv2.putText(frame, alerta, (x,y), font, 1, color, grosor, cv2.LINE_AA)
                frame2 = frame
                linea1 = np.array([[x1,y1],[x2,y2]], np.int32).reshape((-1,1,2))
                frame2 = cv2.polylines(frame2,[linea1],False,(255,0,0),thickness=4)
            
    
    
    
    
    #mostrar la cámara original con el detector y la máscara
    cv2.imshow('Camara principal', frame2)
    cv2.imshow('Borrador de fondo',mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()