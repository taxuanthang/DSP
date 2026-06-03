import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.degradation import extract_texture_features

# ====================================
# CONFIG
# ====================================

DATASET_PATH = "dataset"

CLASSES = [
    "wood",
    "fabric",
    "sand",
    "stone"
]

# ====================================
# LOAD DATASET
# ====================================

X_data = []
y_labels = []

for class_name in CLASSES:

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.exists(class_folder):

        print(
            f"Không tìm thấy thư mục: {class_folder}"
        )

        continue

    for filename in os.listdir(class_folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        image_path = os.path.join(
            class_folder,
            filename
        )

        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:

            print(
                f"Lỗi đọc ảnh: {image_path}"
            )

            continue

        # ====================================
        # FEATURE EXTRACTION
        # ====================================

        feature = extract_texture_features(
            img,
            num_angles=8
        )

        X_data.append(feature)

        y_labels.append(class_name)

# ====================================
# CONVERT TO NUMPY
# ====================================

X_data = np.array(X_data)
y_labels = np.array(y_labels)

print("Số mẫu:", len(X_data))

# ====================================
# TRAIN TEST SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

# ====================================
# TRAIN KNN
# ====================================

knn = KNeighborsClassifier(
    n_neighbors=3
)

knn.fit(
    X_train,
    y_train
)

# ====================================
# PREDICT
# ====================================

y_pred = knn.predict(
    X_test
)

# ====================================
# REPORT
# ====================================

print("\n=== KNN TEXTURE CLASSIFICATION ===\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ====================================
# CONFUSION MATRIX
# ====================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=knn.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=knn.classes_
)

fig, ax = plt.subplots(
    figsize=(7,7)
)

disp.plot(
    cmap="Blues",
    ax=ax
)

ax.set_title(
    "Texture Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "texture_confusion_matrix.png",
    dpi=300
)

plt.show()

# ====================================
# ACCURACY
# ====================================

accuracy = np.mean(
    y_pred == y_test
)

print(
    f"\nAccuracy = {accuracy*100:.2f}%"
)