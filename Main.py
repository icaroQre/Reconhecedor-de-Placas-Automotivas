print("\nIniciando algoritmo de reconhecimento de placas...\n")

import cv2
import numpy as np
import os

# Função para reduzir o tamanho da imagem em 25%
def reduzir_tamanho(img):
    height, width = img.shape[:2]
    new_height, new_width = int(height * 0.75), int(width * 0.75)
    start_row, start_col = (height - new_height) // 2, (width - new_width) // 2
    end_row, end_col = start_row + new_height, start_col + new_width
    return img[start_row:end_row, start_col:end_col]

# Função para preprocessamento da imagem com aumento de contraste
def preprocessar_imagem(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    cl1 = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(cl1, (3, 3), 0)
    equalized = cv2.equalizeHist(blurred)
    kernel = np.ones((3, 3), np.uint8)
    
    # Limiarização adaptativa
    # adaptive_thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Detecção de bordas usando Sobel
    # sobelx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    # sobel = cv2.sqrt(cv2.add(cv2.pow(sobelx, 2), cv2.pow(sobely, 2)))
    
    # Normalizar a imagem para 8 bits
    # sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Operações morfológicas para limpar ruídos
    opened = cv2.morphologyEx(equalized, cv2.MORPH_OPEN, kernel)
    # opened2 = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)
    # opened3 = cv2.morphologyEx(opened2, cv2.MORPH_ERODE, kernel)
    # opened4 = cv2.morphologyEx(opened3, cv2.MORPH_DILATE, kernel)
    dimmed = cv2.convertScaleAbs(opened, alpha=0.5, beta=0)
    
    return dimmed

# Função para carregar imagens do diretório
def carregar_imagens(diretorio, num_imagens=30):
    images = []
    for i in range(num_imagens):
        image_path = os.path.join(diretorio, f'car_{i+1}.jpg')
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Erro ao carregar a imagem {image_path}")
        else:
            print(f"Imagem {image_path} não encontrada")
    return images

# Função principal
def main():
    # Carregar o classificador Haar Cascade para placas de veículos
    plateCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

    # Carregar imagens do diretório
    image_dir = 'images'
    images = carregar_imagens(image_dir, num_imagens=30)
    
    if not images:
        print("Nenhuma imagem carregada")
        return

    # Processar cada imagem
    for idx, img in enumerate(images):
        reduced_img = reduzir_tamanho(img)
        preprocessed_img = preprocessar_imagem(reduced_img)
        plates = plateCascade.detectMultiScale(preprocessed_img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        print(f"Imagem {idx+1} - Placas detectadas: {plates}")
        for i, (x, y, w, h) in enumerate(plates):
            cv2.rectangle(reduced_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            plate_img = reduced_img[y: y + h, x: x + w]
            cv2.imshow(f"Imagem {idx+1} - Placa {i+1}", plate_img)
        
        # Mostrar a imagem original com as placas detectadas
        # cv2.imshow(f"Imagem {idx+1}", reduced_img)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
