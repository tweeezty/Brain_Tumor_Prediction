🧠 Brain Tumor Detection using Deep Learning (PyTorch)
This project implements a Convolutional Neural Network (CNN) to detect brain tumors from MRI images using PyTorch.
The model classifies MRI scans into two categories:
1)Tumor
2)No Tumor

📌 Project Description
Brain tumor detection is a critical task in medical imaging.
This project uses deep learning and computer vision techniques to automatically analyze MRI images and predict whether a brain tumor is present.

The model is trained locally on a labeled MRI dataset and can be used to make predictions on new images.

🛠️ Technologies Used
Programming Language: Python
Framework: PyTorch
Libraries:
TorchVision
OpenCV
NumPy
Matplotlib
Model: Convolutional Neural Network (CNN)

📂 Project Structure
Brain Tumor Detection/
│
├── dataset/
│   ├── train/
│   │   ├── tumor/
│   │   └── no_tumor/
│   └── test/
│       ├── tumor/
│       └── no_tumor/
│
├── model.py
├── train.py
├── predict.py
├── brain_tumor_model.pth
├── requirements.txt
├── README.md
└── venv/
⚠️ Note: The dataset/ folder and venv/ are not included in the GitHub repository due to size limitations.

📊 Dataset
Dataset Link
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
MRI brain scan images
Binary classification:
   1)tumor
   2)no_tumor
Dataset is organized into train and test folders

📌 You can download similar datasets from Kaggle.

⚙️ Installation & Setup
1️⃣ Clone the repository
  git clone https://github.com/your-username/brain-tumor-detection.git
  cd brain-tumor-detection
2️⃣ Create a virtual environment
  python -m venv venv
3️⃣ Activate the virtual environment
Windows (PowerShell):

  .\venv\Scripts\Activate
4️⃣ Install required dependencies
   pip install -r requirements.txt
🏋️ Model Training
To train the model on the dataset:
   python train.py
   
After training, the model is saved as:
   brain_tumor_model.pth
🔍 Model Prediction
To predict on a new MRI image:
Place an MRI image in the project directory
Rename it to sample.jpg
Run:
   python predict.py
Output:
🧠 Brain Tumor Detected
or

✅ No Brain Tumor
📈 Results
The CNN learns features from MRI images
Training loss decreases over epochs
The model performs effective binary classification
🚀 Future Enhancements
Multi-class classification (Glioma, Meningioma, Pituitary)
Transfer learning using ResNet / VGG
Web deployment using Streamlit
Improved evaluation metrics and visualization

👤 Author
Mohamed Dhaha
Machine Learning & Deep Learning Enthusiast

⭐ Acknowledgements
PyTorch Official Documentation

Medical imaging datasets from Kaggle

