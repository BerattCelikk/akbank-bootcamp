# Brain Tumor Classification with CNN & Grad-CAM

Kaggle = https://www.kaggle.com/code/beraterolelk/brain-tumor-classification-with-cnn-grad-cam


## 📋 Proje Hakkında
Bu proje, beyin tümörlerini **glioma, meningioma, pituitary ve no tumor** olmak üzere 4 sınıfta sınıflandırmak için **Convolutional Neural Network (CNN)** kullanmaktadır. Ayrıca modelin hangi bölgeleri dikkate aldığını görselleştirmek için **Grad-CAM** yöntemi uygulanmıştır.

---

## 🛠️ Kullanılan Kütüphaneler
- `os`, `random` → Dosya işlemleri ve rastgele seçimler
- `numpy` → Sayısal hesaplamalar
- `matplotlib` → Görselleştirme
- `tensorflow.keras` → Derin öğrenme ve CNN modelleme
- `sklearn.metrics` → Confusion Matrix ve Classification Report
- `seaborn` → Confusion Matrix heatmap görselleştirme
- `cv2` → OpenCV ile görüntü işleme

---

## 📁 Veri Seti
- **Training:** `Training/` klasörü altında dört sınıf  
- **Testing:** `Testing/` klasörü altında dört sınıf  

Sınıflar:
- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

---

## 🔄 Veri Önişleme ve Augmentation
- Görüntüler 224x224 boyutuna yeniden boyutlandırıldı
- Piksel değerleri [0,1] aralığına normalize edildi
- Data augmentation ile:
  - ±15° döndürme
  - %10 yatay/dikey kaydırma
  - %10 zoom
  - Shear transform
  - Yatay çevirme
- Validation split: %20

---

## 🧠 CNN Modeli
**Katmanlar:**
1. Conv2D (32 filtre, 3x3, ReLU) → MaxPooling2D
2. Conv2D (64 filtre, 3x3, ReLU) → MaxPooling2D
3. Conv2D (128 filtre, 3x3, ReLU) → MaxPooling2D
4. Flatten
5. Dense (128 nöron, ReLU)
6. Dropout (%50)
7. Dense (4 nöron, softmax) → 4 sınıf için çıkış

**Optimizasyon ve Kayıp:**
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Metric: Accuracy

**Eğitim:**
- Epoch: 20
- Batch Size: 32

---

## 📊 Eğitim Sonuçları
- **En yüksek doğruluk (validation):** ~81%
- **Test set doğruluk:** ~86%
- Eğitim ve doğrulama kaybı ve doğruluk grafikleri matplotlib ile çizildi

---

## 🧾 Confusion Matrix & Classification Report
- Confusion Matrix ile gerçek vs tahmin edilen sınıflar karşılaştırıldı
- Her sınıf için precision, recall, f1-score hesaplandı
- Heatmap seaborn ile görselleştirildi

Örnek Confusion Matrix:
[[241 41 1 17]
[ 17 196 61 32]
[ 2 1 402 0]
[ 3 2 0 295]]


---

## 🔥 Grad-CAM
- Modelin hangi bölgeleri dikkate aldığını görselleştirmek için Grad-CAM kullanıldı
- Son Conv2D katmanı: `conv2d_5`
- Heatmap oluşturuldu ve orijinal görüntü üzerine süperimpose edildi
- Görselleştirme OpenCV ve matplotlib ile yapıldı

**Adımlar:**
1. Test görüntüsü 1xHxWxC formatına çevrildi
2. Grad-CAM heatmap hesaplandı
3. Heatmap yeniden boyutlandırıldı ve Jet colormap ile renklendirildi
4. Orijinal görüntü ile birleştirildi (%40 heatmap, %60 orijinal)
5. Sonuç matplotlib ile gösterildi

---

## 📌 Kullanım Örneği
```python
# Test görüntüsünü al
img, label = test_gen[0][0][0], test_gen[0][1][0]
img_array = np.expand_dims(img, axis=0)

# Grad-CAM heatmap oluştur
heatmap = get_grad_cam_heatmap_seq(model, img_array, last_conv_layer)

# Heatmap'i orijinal görüntü ile birleştir
img_rgb = img.astype(np.uint8)
heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
superimposed_img = heatmap_colored * 0.4 + img_rgb

# Görselleştir
plt.imshow(superimposed_img.astype(np.uint8))
plt.axis('off')
plt.show()
