import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from src.degradation import extract_texture_features

# --- MÔ PHỎNG DỮ LIỆU ĐỂ CHẠY THỬ ---
np.random.seed(42)
num_samples_per_class = 20
X_data = []
y_labels = []

for class_idx, class_name in enumerate(['Wood', 'Fabric', 'Sand', 'Stone']):
    for _ in range(num_samples_per_class):
        base_feature = np.zeros(8)
        if class_idx == 0: base_feature[2] = 0.6  
        elif class_idx == 1: base_feature[5] = 0.7 
        elif class_idx == 2: base_feature = np.ones(8) * 0.125 
        else: base_feature[[0, 4]] = 0.4          
        
        feature = base_feature + np.random.uniform(0, 0.1, 8)
        feature /= np.sum(feature)
        
        X_data.append(feature)
        y_labels.append(class_name)

X_data = np.array(X_data)
y_labels = np.array(y_labels)

# Chia Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Huấn luyện KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Dự đoán
y_pred = knn.predict(X_test)

# In báo cáo và vẽ ma trận nhầm lẫn
print("--- ĐÁNH GIÁ MÔ HÌNH KNN ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax)
ax.set_title("Ma trận nhầm lẫn - Phân loại Kết cấu (KNN)")

plt.savefig("texture_confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.show()