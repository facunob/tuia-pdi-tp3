import cv2
import numpy as np
import sys

# Dilatacion
kernel_dilatacion = np.ones((9, 9), np.uint8)
kernel_dilatacion_soft = np.ones((3, 3), np.uint8)

def dilate_img(img, soft=False):
  kernel = kernel_dilatacion_soft if soft else kernel_dilatacion
  return cv2.dilate(img, kernel)


#Filtrado HSV (rojos)
lower_rojo = np.array([0, 150, 100])
upper_rojo = np.array([10, 255, 255])

def filtrar_rojos(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  return cv2.inRange(hsv, lower_rojo, upper_rojo)


# Dibujar numeros en frame
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontColor              = (255,255,255)
fontScale              = 0.5
thickness              = 1
lineType               = 2

def draw_number(frame, position, number):
  cv2.putText(frame, f'{number}',
        position,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType
  )

# Buscar Dados, dibujar rectangulo. Buscar numero de puntos
def draw_dados(frame):
    mask = filtrar_rojos(frame)
    mask = dilate_img(mask)
    img = cv2.bitwise_and(frame, frame,  mask=mask)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (
        componentes_conectadas_dados,
        etiquetas_dados,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(mask, cv2.CV_32S, connectivity=8)
    if(componentes_conectadas_dados != 6 ):
      return;

    for i in range(1, componentes_conectadas_dados):
      mascara = np.uint8(etiquetas_dados == i)
      dado = cv2.bitwise_and(img_gray, img_gray, mask=mascara)
      _, puntos_blancos = cv2.threshold(dado, 190, 255, cv2.THRESH_BINARY)
      puntos_blancos = dilate_img(puntos_blancos, soft=True)

      count_puntos = 0
      ( componentes_puntos, etiquetas_puntos, estadisticas, _, ) = cv2.connectedComponentsWithStats(puntos_blancos, cv2.CV_32S, connectivity=8)
      for j in range(1, componentes_puntos):
        mascara_punto = np.uint8(etiquetas_puntos == j)
        area = estadisticas[j, cv2.CC_STAT_AREA]
        if area > 10:
          count_puntos +=1
      if (count_puntos == 0):
        return;
    
      x = stats[i, cv2.CC_STAT_LEFT]
      y = stats[i, cv2.CC_STAT_TOP]
      w = stats[i, cv2.CC_STAT_WIDTH]
      h = stats[i, cv2.CC_STAT_HEIGHT]
      cv2.rectangle(frame, (x, y), (x+w,y+h), 150, 1)
      draw_number(frame, (x+w, y+h), count_puntos)


# Funcion para detectar cambios en 2 frames de un video
def calculate_frame_diff(prev_frame, current_frame):
    diff_frame = cv2.absdiff(prev_frame, current_frame)
    gray_diff_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
    _, threshold_diff = cv2.threshold(gray_diff_frame, 30, 255, cv2.THRESH_BINARY)
    return threshold_diff


# Main: leer y grabar video
def main(path):
  cap = cv2.VideoCapture(f'{path}.mp4')
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  out = cv2.VideoWriter(f'{path}_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width/3),int(height/3)))

  ret, prev_frame = cap.read()
  ret, current_frame = cap.read()

  while ret:
      diff = calculate_frame_diff(prev_frame, current_frame)

      frame = cv2.resize(current_frame, dsize=(int(width/3), int(height/3)))
      if cv2.countNonZero(diff) < 2000:
          draw_dados(frame)

      out.write(frame)
      prev_frame = current_frame
      ret, current_frame = cap.read()
  print(f'Se creo el video {path}.mp4')

  # Fin
  cap.release()
  out.release()
  cv2.destroyAllWindows()



if __name__ == "__main__":
    assert len(sys.argv) > 1, "INGRESE EL ARCHIVO COMO PARAMETRO!"
    main(sys.argv[-1])
