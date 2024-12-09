import tkinter as tk
from tkinter import Canvas, messagebox, Label
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from pyscreenshot import grab

# Initialize main window
window = tk.Tk()
window.title("Nhận diện chữ viết tay bằng CNN")
window.geometry("800x680")
window.resizable(0, 0)

canvas1 = Canvas(window, width=500, height=250, bg='ivory')
canvas1.place(x=5, y=120)

l1 = tk.Label(canvas1, text="Nhập và tạo", font=('Times New Roman', 20))
l1.place(x=5, y=0)

t1 = tk.Entry(canvas1, width=20, border=5)
t1.place(x=150, y=0)

# Paths
dataset_path = r"dataset.npy"
labels_path = r"labels.npy"
model_path = r"digit_recognizer_cnn.h5"

# Functions for dataset generation and training
def generate_dataset():
    digit_label = t1.get()
    if not digit_label.isdigit():
        messagebox.showerror("Lỗi", "Vui lòng nhập một chữ số hợp lệ.")
        return

    digit_label = int(digit_label)
    images_folder = "captured_images"
    os.makedirs(images_folder, exist_ok=True)
    digit_folder = os.path.join(images_folder, str(digit_label))
    os.makedirs(digit_folder, exist_ok=True)

    for i in range(5):
        img = grab(bbox=(50, 200, 411, 482))
        img_path = os.path.join(digit_folder, f"{i}.png")
        img.save(img_path)

    messagebox.showinfo("Kết quả", "Tạo dữ liệu thành công!")

b1 = tk.Button(canvas1, text="1. Tạo dữ liệu", font=('Times New Roman', 15), bg="orange", command=generate_dataset)
b1.place(x=5, y=50)

def prepare_data():
    images_folder = "captured_images"
    data, labels = [], []

    for label in os.listdir(images_folder):
        label_folder = os.path.join(images_folder, label)
        for img_path in os.listdir(label_folder):
            img = cv2.imread(os.path.join(label_folder, img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            data.append(img)
            labels.append(int(label))

    data = np.array(data).reshape(-1, 28, 28, 1) / 255.0
    labels = to_categorical(labels, 10)

    np.save(dataset_path, data)
    np.save(labels_path, labels)
    messagebox.showinfo("Kết quả", "Dữ liệu đã được chuẩn bị.")

b2 = tk.Button(canvas1, text="2. Chuẩn bị dữ liệu", font=('Times New Roman', 15), bg="pink", command=prepare_data)
b2.place(x=5, y=100)

def train_model():
    data = np.load(dataset_path)
    labels = np.load(labels_path)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(data, labels, epochs=10, validation_split=0.2, batch_size=32)
    model.save(model_path)

    messagebox.showinfo("Kết quả", "Huấn luyện mô hình thành công!")

b3 = tk.Button(canvas1, text="3. Huấn luyện mô hình", font=('Times New Roman', 15), bg="green", command=train_model)
b3.place(x=5, y=150)

def predict_digit():
    if not os.path.exists(model_path):
        messagebox.showerror("Lỗi", "Không tìm thấy mô hình. Vui lòng huấn luyện mô hình trước.")
        return

    img = grab(bbox=(130, 500, 500, 700))
    img.save("temp.png")

    img = cv2.imread("temp.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1) / 255.0

    model = load_model(model_path)
    prediction = np.argmax(model.predict(img), axis=-1)[0]

    messagebox.showinfo("Kết quả", f"Chữ số dự đoán: {prediction}")

b4 = tk.Button(canvas1, text="4. Dự đoán", font=('Times New Roman', 15), bg="white", command=predict_digit)
b4.place(x=5, y=200)

# Canvas for drawing
canvas2 = Canvas(window, width=500, height=250, bg='black')
canvas2.place(x=5, y=380)

lastx, lasty = None, None

def activate_paint(e):
    global lastx, lasty
    lastx, lasty = e.x, e.y
    canvas2.bind('<B1-Motion>', paint)

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    canvas2.create_line((lastx, lasty, x, y), width=10, fill='white')
    lastx, lasty = x, y

canvas2.bind('<1>', activate_paint)

def clear_canvas():
    canvas2.delete("all")

clear_btn = tk.Button(canvas2, text="Clear", fg="white", bg="red", command=clear_canvas)
clear_btn.place(x=0, y=0)

window.mainloop()
