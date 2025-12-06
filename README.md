##  README.md — Fruits Classification using CNN (Fruits-360 Dataset)

```markdown
#  Fruits Classification using Convolutional Neural Network

This project is a Deep Learning based Fruits Classification system trained using the **Fruits-360** dataset.  
The model uses a **custom CNN architecture** and is trained on Kaggle GPU (P100).

---

##  Project Overview

Fruit classification can be used in:

- Smart agriculture
- Food quality analysis
- Supermarket automation
- Computer vision-based IoT systems

In this project, we classify **120+ fruit categories** with high accuracy using deep learning.

---

##  Dataset

 Dataset: Fruits-360  
 Source: Kaggle Public Dataset  
 Structure:

```

fruits-360/
├── Training/
└── Test/

````

Images are **100×100 RGB**, well-organized into subfolders per class.

---

## 🔧 Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python | Implementation |
| TensorFlow / Keras | Neural Network |
| ImageDataGenerator | Data loading & augmentation |
| GPU P100 (Kaggle) | Fast training |
| Matplotlib / NumPy | Visualization & math |

---

##  Model Architecture (Custom CNN)

- Conv2D + ReLU + MaxPooling layers (4 blocks)
- Dropout for regularization
- Dense layer + Softmax output
- Optimizer: Adam
- Loss: Categorical Crossentropy

> Achieved **~97% Test Accuracy**

---

##  Training Details

- Epochs: 20
- Batch size: 64
- Input size: 100×100×3
- GPU Accelerator: Tesla P100 (Kaggle)
- Data Augmentation applied

---

##  Model Files

After training, the following are generated:

| File | Description |
|------|-------------|
| `final_model.h5` | Final saved model |
| `final_model.keras` | Keras native format model |
| `best_model.h5` | Best accuracy checkpoint |
| `epoch_xx.h5` | Models saved every epoch |

> These files are stored inside `/kaggle/working` during training.

---

##  Inference Example

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("final_model.h5")

img_path = "sample_fruit.jpg"
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
class_idx = np.argmax(pred)
print("Predicted Class:", class_idx)
````

---

## � Results Visualization

* Accuracy and loss curves for training & validation
* Random prediction visualization with correct/incorrect classification colors

---

## To Run Notebook on Kaggle

1. Upload / attach dataset: **Fruits-360**
2. Select accelerator: **GPU P100**
3. Run all notebook cells sequentially
4. Download saved model from the right panel → *Output Files*

---

## Future Improvements

* Transfer learning using MobileNetV2/EfficientNet
* Real-time webcam fruit recognition
* Convert to TensorFlow Lite for mobile apps
* Deploy using Flask/Streamlit

```
