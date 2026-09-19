# Pneumonia Detection Model Using Deep Learning

This project uses transfer learning with DenseNet121 to classify chest X-ray images into two categories:

- Normal
- Pneumonia

## Technologies Used

- Python
- TensorFlow
- Keras
- DenseNet121
- Flask
- HTML
- CSS
- NumPy

## Project Workflow

1. Collect and prepare chest X-ray images.
2. Resize images to 128 × 128 pixels.
3. Apply data augmentation.
4. Convert grayscale images to three channels.
5. Use DenseNet121 pretrained on ImageNet.
6. Freeze the base model during initial training.
7. Add Global Average Pooling, Dropout, and a sigmoid output layer.
8. Save the trained model.
9. Use Flask to create a web interface.

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

Place your dataset in the expected folders described in `model/train_model.py`, then run:

```bash
python model/train_model.py
```

The trained model will be saved as:

```text
pneumonia_model.keras
```

## Run the Flask Application

```bash
python app.py
```

Open the local address shown in your terminal.

## Important Note

This project is for educational and demonstration purposes only. It must not be used as a substitute for professional medical diagnosis.

## Author

Tamal Maitra
