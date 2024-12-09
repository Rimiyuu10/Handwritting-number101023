import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import pandas as pd
import pyscreenshot as ImageGrab
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Màn hình chính
window = tk.Tk()
window.title("Nhận diện chữ viết tay (CNN)")
window.resizable(0, 0)

# Canvas Setup
canvas1 = Canvas(window, width=500, height=250, bg='ivory')
canvas1.place(x=5, y=120)

# Ký tự nhập vào và dataset
l1 = tk.Label(canvas1, text="Nhập và tạo", font=('Times New Roman', 20))
l1.place(x=5, y=0)

t1 = tk.Entry(canvas1, width=20, border=5)
t1.place(x=150, y=0)


BASE_PATH = r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG"
IMAGES_FOLDER = os.path.join(BASE_PATH, "captured_images")
DATASET_PATH = os.path.join(BASE_PATH, "dataset.csv")
MODEL_PATH = os.path.join(BASE_PATH, "model", "digit_recognizer_cnn.h5")

def screen_capture():
    digit_label = t1.get()
    digit_folder = os.path.join(IMAGES_FOLDER, digit_label)
    os.makedirs(digit_folder, exist_ok=True)

    paint_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Accessories\Paint"
    os.startfile(paint_path)

    import time
    time.sleep(15)

    for i in range(0, 5):
        time.sleep(8)
        im = ImageGrab.grab(bbox=(50, 200, 411, 482))
        im.save(os.path.join(digit_folder, f"{i}.png"))
        print(f"Đã lưu: {digit_folder}/{i}.png")
        print("Xóa màn hình và vẽ lại.")

    messagebox.showinfo("Result", "Chụp màn hình đã xong !!")

def generate_dataset():
    header = ["label"] + [f"pixel{i}" for i in range(784)]
    with open(DATASET_PATH, 'w') as f:
        f.write(','.join(header) + '\n')

    for label in range(10):
        dirList = glob.glob(f"{IMAGES_FOLDER}/{label}/*.png")
        for img_path in dirList:
            im = cv2.imread(img_path)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
            roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

            data = [label]
            for row in roi:
                data.extend([1 if pixel > 100 else 0 for pixel in row])

            with open(DATASET_PATH, 'a') as f:
                f.write(','.join(map(str, data)) + '\n')

    messagebox.showinfo("Kết quả", "Tạo tập dữ liệu hoàn thành!!!")

def train_save_accuracy():
    #Đoc data
    data = pd.read_csv(DATASET_PATH)

    X = data.drop("label", axis=1).values
    y = data["label"].values

    X = X.reshape(-1, 28, 28, 1) / 255.0
    y = to_categorical(y, num_classes=10)


    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)


    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])


    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(train_x, train_y,
                        validation_split=0.2,
                        epochs=10,
                        batch_size=32)

    test_loss, test_accuracy = model.evaluate(test_x, test_y)


    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    messagebox.showinfo("Kết quả", f"Độ chính xác của mô hình là: {test_accuracy*100:.2f}%")

def prediction():

    model = tf.keras.models.load_model(MODEL_PATH)

    img = ImageGrab.grab(bbox=(130, 500, 500, 700))
    img.save("paint.png")


    im = cv2.imread("paint.png")
    load = Image.open("paint.png")
    load = load.resize((280, 280))
    photo = ImageTk.PhotoImage(load)


    img_label = Label(canvas3, image=photo, width=280, height=280)
    img_label.image = photo
    img_label.place(x=0, y=0)


    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
    _, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)


    X = roi.reshape(1, 28, 28, 1) / 255.0

    predictions = model.predict(X)
    predicted_digit = np.argmax(predictions)


    a1 = tk.Label(canvas3, text="Predictions=  ", font=("Times New Roman", 20))
    a1.place(x=5, y=350)

    b1 = tk.Label(canvas3, text=str(predicted_digit), font=("Times New Roman", 20))
    b1.place(x=200, y=350)

# nút Setup
b1 = tk.Button(canvas1, text="1. Mở pain và tạo hình dữ liệu",
               font=('Times New Roman', 15), bg="orange", fg="black", command=screen_capture)
b1.place(x=5, y=50)

b2 = tk.Button(canvas1, text="2. Tạo tập dữ liệu", font=('Times New Roman', 15),
               bg="pink", fg="blue", command=generate_dataset)
b2.place(x=5, y=100)

b3 = tk.Button(canvas1, text="3. Huấn luyện mô hình CNN",
               font=('Times New Roman', 15), bg="green", fg="white", command=train_save_accuracy)
b3.place(x=5, y=150)

b4 = tk.Button(canvas1, text="4. Dự đoán trực tiếp", font=('Times New Roman', 15),
               bg="white", fg="red", command=prediction)
b4.place(x=5, y=200)

# vẽ canvas
canvas2 = Canvas(window, width=500, height=250, bg='black')
canvas2.place(x=5, y=380)

def activate_paint(e):
    global lastx, lasty
    canvas2.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y

def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    canvas2.create_line((lastx, lasty, x, y), width=40, fill=('white'))
    lastx, lasty = x, y

canvas2.bind('<1>', activate_paint)

def clear():
    canvas2.delete("all")

btn = tk.Button(canvas2, text="clear", fg="white", bg="green", command=clear)
btn.place(x=0, y=0)

#
canvas3 = Canvas(window, width=280, height=530, bg='green')
canvas3.place(x=500, y=120)

# man hinh config
window.geometry("800x680")
window.mainloop()
