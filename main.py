import cv2
from scipy.spatial import distance
import os
import mediapipe as mp

app_folder = os.path.dirname(os.path.abspath(__file__))

def calc_euclidean_distance(dicionario):
    distancias = {}
    for key in dicionario.keys():
        for point in dicionario.keys():
            if point!=key and (point, key) not in dicionario: #sabendo que a distancia é a mesma de um pro outro
                distancias[(point, key)] = distance.euclidean(dicionario[key], dicionario[point])
                distancias[(key, point)] = distance.euclidean(dicionario[key], dicionario[point])
    return distancias

def runner(path, mp_drawing, mp_holistic):
    print('ge')
    video = cv2.VideoCapture(path)
    cont=0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while video.isOpened():
            ret, frame = video.read()

            if not ret or cv2.waitKey(1) == ord('q'):
                print("Can't receive frame (stream end?). Exiting ...")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = holistic.process(image)
            #print(results.face_landmarks) acessamos os pontos da face
            #results.pose_landmarks acessamos os pontos do corpo
            
            if results.right_hand_landmarks!=None:
                #with open(app_folder+'/results/right_hand.txt', 'a') as txt:
                 #   for element in results.right_hand_landmarks.landmark:
                  #      txt.write(str(element))
                right_hand_landmarks = {lmk: [results.right_hand_landmarks.landmark[lmk].x, results.right_hand_landmarks.landmark[lmk].y, results.right_hand_landmarks.landmark[lmk].z] for lmk in range(len(results.right_hand_landmarks.landmark))}
                print(results.right_hand_landmarks)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('test', image) 
            '''   
                #print(results.pose_landmarks.landmark[0])
            #recolorindo a imagem para rgb
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.imread('douglas.jpg')

            #desenhando as marcas faciais
            mp_drawing.draw_landmarks(
                image, 
                results.face_landmarks, 
                mp_holistic.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )

            #Mão direita
            

            #Mão esquerda
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            #Pose Detection
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            cv2.imshow('Holistic Model Detections', image)''' 
    video.release()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
runner('BOMDIA.avi', mp_drawing, mp_holistic)
cv2.destroyAllWindows()