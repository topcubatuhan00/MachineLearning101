import cv2
from matplotlib import pyplot as plt
import numpy as np

img_path = "./m.png"
img = cv2.imread(img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, output2) = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
output2 = cv2.GaussianBlur(output2, (5, 5), 1)
output2 = cv2.Canny(output2, 180, 255)
plt.imshow(output2, cmap=plt.get_cmap("gray"))

circles = cv2.HoughCircles(output2, cv2.HOUGH_GRADIENT, 1, 10, param1=180, param2=27, minRadius=20, maxRadius=60)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        center = (i[0], i[1])  # Center as a tuple
        cv2.circle(img, center, i[2], (0, 255, 0), 2)
        cv2.circle(img, center, 2, (0, 0, 255), 3)

    plt.imshow(img)
else:
    print("No circles found in the image.")

plt.show()


"""
Bu kod, bir resim dosyasından halkaları algılamak ve çizmek için OpenCV (Open Source Computer Vision) kütüphanesini kullanır. İşlemlerin her birini aşağıda daha ayrıntılı olarak açıklıyorum:

1. `cv2.imread` ile Resim Yükleme:
   - İlk olarak, belirtilen dosya yolu (`img_path`) kullanılarak bir resim dosyası (`m.png`) yüklenir. Bu yükleme, bir NumPy dizisi olarak temsil edilen bir görüntü döndürür.

2. Griye Dönüştürme:
   - Resim, `cv2.cvtColor` fonksiyonu kullanılarak BGR (Mavi, Yeşil, Kırmızı) renk uzayından Gri renk uzayına dönüştürülür. Bu, daha sonra kenar tespiti ve Hough dönüşümü gibi işlemleri gerçekleştirmek için kullanılacak gri bir görüntü oluşturur.

3. Eşikleme:
   - `cv2.threshold` işlevi ile gri görüntüde eşikleme işlemi yapılır. Eşik değeri 120 olarak ayarlanmıştır. Bu işlem, görüntüyü siyah-beyaz hale getirir, böylece kenar tespiti daha etkili hale gelir.

4. Gauss Bulanıklığı:
   - `cv2.GaussianBlur` ile görüntüye Gauss bulanıklığı uygulanır. Bu işlem, kenar tespiti işlemini daha hassas hale getirir ve gürültüyü azaltır.

5. Kenar Tespiti:
   - `cv2.Canny` işlevi ile görüntü üzerinde kenar tespiti yapılır. Kenarlar beyaz, arka plan siyah olarak vurgulanır.

6. Hough Dönüşümü ile Halka Algılama:
   - `cv2.HoughCircles` işlevi kullanılarak halkalar görüntüde algılanır. Bu işlev, belirli bir parametre seti ile dairesel nesneleri tespit etmek için Hough dönüşümünü kullanır. Algılanan halkalar `circles` değişkenine atanır.

7. Halkaları Çizme:
   - Eğer halkalar algılandıysa (yani, `circles` değişkeni `None` değilse), bu halkalar `for` döngüsü ile gezilir.
   - Halkalar `cv2.circle` işlevi kullanılarak orijinal görüntü üzerine çizilir. Halkaların merkezi yeşil, kenarları ise kırmızı renkte çizilir.

8. Sonuç Görüntüsünü Gösterme:
   - Son olarak, çizilen halkaların eklenmiş olduğu orijinal görüntü, `plt.imshow` ile görselleştirilir.

Eğer hiç halka algılanmazsa, "No circles found in the image." mesajı ekrana yazdırılır.
"""