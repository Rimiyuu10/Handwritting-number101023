import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

window=tk.Tk()
window.title("Nhận diện chữ viết tay")
window.resizable(0,0)

load1 =Image.open("Handwr2.png")
photo1=ImageTk.PhotoImage(load1)

header=tk.Button(window,bg="pink",image=photo1)
header.place(x=5,y=0)

header = tk.Button(window, bg="pink", image=photo1, command=window.destroy)
header.place(x=5, y=0)


canvas1 = Canvas(window,width=500,height=250,bg='ivory')
canvas1.place(x=5,y=120)

import tkinter as tk
from tkinter import *
from tkinter import messagebox
import os



l1 = tk.Label(canvas1, text="Nhập và tạo", font=('Times New Roman', 20))
l1.place(x=5, y=0)

t1 = tk.Entry(canvas1, width=20, border=5)
t1.place(x=150, y=0)


def screen_capture():
    import pyscreenshot as ImageGrab
    import time

    paint_path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Accessories\Paint"
    images_folder = r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG\captured_images"


    #tao thu muc
    digit_label = t1.get()
    digit_folder = os.path.join(images_folder, digit_label)
    os.makedirs(digit_folder, exist_ok=True)

    # Mở pain
    os.startfile(paint_path)
    time.sleep(15)

    # Cap màn hình
    for i in range(0, 5):
        time.sleep(8)
        im = ImageGrab.grab(bbox=(50, 200, 411, 482))  # Điều chỉnh khung theo pixel
        im.save(os.path.join(digit_folder, f"{i}.png"))
        print(f"Đã lưu: {digit_folder}/{i}.png")
        print("Xóa màn hình và vẽ lại.")

    messagebox.showinfo("Result", "Chụp màn hình đã xong !!")


b1 = tk.Button(canvas1, text="1. Mở pain và tạo hình dữ liệu",
               font=('Times New Roman', 15), bg="orange", fg="black", command=screen_capture)
b1.place(x=5, y=50)


def generate_dataset():
    import cv2
    import csv
    import glob

    dataset_path = r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG\dataset.csv"
    images_folder = r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG\captured_images"

    header = ["label"] + [f"pixel{i}" for i in range(784)]
    with open(dataset_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for label in range(10):
        dirList = glob.glob(f"{images_folder}/{label}/*.png")
        for img_path in dirList:
            im = cv2.imread(img_path)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
            roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

            data = [label]
            for row in roi:
                data.extend([1 if pixel > 100 else 0 for pixel in row])

            with open(dataset_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)

    messagebox.showinfo("Kết quả", "Tạo tập dữ liệu hoàn thành!!!")


b2 = tk.Button(canvas1, text="2. Tạo tập dữ liệu", font=('Times New Roman', 15),
               bg="pink", fg="blue", command=generate_dataset)
b2.place(x=5, y=100)


def train_save_accuracy():
    import pandas as pd
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn import metrics
    import joblib

    dataset_path = r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG\dataset.csv"
    model_path = r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG\model\digit_recognizer.pkl"

    data = pd.read_csv(dataset_path)
    data = shuffle(data)
    X = data.drop(["label"], axis=1)
    Y = data["label"]
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

    classifier = SVC(kernel="linear", random_state=6)
    classifier.fit(train_x, train_y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(classifier, model_path)

    prediction = classifier.predict(test_x)
    acc = metrics.accuracy_score(prediction, test_y)
    messagebox.showinfo("Kết quả", f"Độ chính xác của là{acc}")


b3 = tk.Button(canvas1, text="3. Huấn luyện mô hình, lưu mô hình và tính toán độ chính xác",
               font=('Times New Roman', 15), bg="green", fg="white", command=train_save_accuracy)
b3.place(x=5, y=150)


def prediction():
    import joblib
    import cv2
    import numpy as np  # pip install numpy
    import time
    import pyscreenshot as ImageGrab
    import os
    from PIL import Image

    model = joblib.load(r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG\model\digit_recognizer.pkl")

    # Chụp ảnh màn hình
    img = ImageGrab.grab(bbox=(130, 500, 500, 700))
    img.save("paint.png")

    # Xử lý ảnh
    im = cv2.imread("paint.png")
    load = Image.open("paint.png")
    load = load.resize((280, 280))  # Resize để hiển thị trên canvas
    photo = ImageTk.PhotoImage(load)

    # Hiển thị ảnh lên canvas
    img_label = Label(canvas3, image=photo, width=280, height=280)
    img_label.image = photo  # Giữ tham chiếu đến ảnh
    img_label.place(x=0, y=0)  # Đặt vị trí cho ảnh

    # Chuyển ảnh sang ảnh grayscale và làm mờ
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

    # Threshold hóa ảnh
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

    rows, cols = roi.shape

    X = []


    # Điền dữ liệu vào mảng X với các pixel
    for i in range(rows):
        for j in range(cols):
            k = roi[i, j]
            if k > 100:
                k = 1
            else:
                k = 0
            X.append(k)

    # Dự đoán
    predictions = model.predict([X])

    # Hiển thị kết quả dự đoán
    a1 = tk.Label(canvas3, text="Predictions=  ", font=("Times New Roman", 20))
    a1.place(x=5, y=350)

    b1 = tk.Label(canvas3, text=predictions[0], font=("Times New Roman", 20))
    b1.place(x=200, y=350)




# Tạo nút dự đoán Tkinter
b4 = tk.Button(canvas1, text="4. Dự đoán trực tiếp", font=('Times New Roman', 15),
               bg="white", fg="red", command=prediction)
b4.place(x=5, y=200)








canvas2 = Canvas(window,width=500,height=250,bg='black')
canvas2.place(x=5,y=380)


def activate_paint(e) :
    global lastx, lasty
    canvas2.bind('<B1-Motion>',paint)
    lastx,lasty= e.x, e.y

def paint(e) :
    global lastx,lasty
    x,y =e.x,e.y
    canvas2.create_line((lastx,lasty,x,y),width=40,fill=('white'))
    lastx,lasty =x,y

canvas2.bind('<1>', activate_paint)

def clear () :
    canvas2.delete("all")

btn=tk.Button(canvas2,text="clear",fg="white",bg="green",command=clear)
btn.place(x=0,y=0)







def livepaint():

    import joblib
    import cv2
    import numpy as np
    import time
    import pyscreenshot as ImageGrab
    import os
    from tkinter import messagebox
    canvas2.delete("all")

    # Mở Paint để cho phép người dùng vẽ và tiến hành dự đoán
    os.startfile("C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Accessories/Paint")

    model_path = r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG\model\digit_recognizer.pkl"
    img_folder = r"C:\Users\admin\Downloads\code py\XLAVTG\XLAVTG\img"
    os.makedirs(img_folder, exist_ok=True)

    if not os.path.exists(model_path):
        messagebox.showerror("Lỗi", "KHông tìm thấy mô hình hãy huấn luyện trước.")
        return

    # Load mô hình đã được huấn luyện
    model = joblib.load(model_path)
    time.sleep(15)  # Give user 15 seconds to draw in Paint

    while True:
        # Chụp khu vực người dùng sẽ vẽ
        img = ImageGrab.grab(bbox=(60, 170, 400, 550))  # Điều chỉnh kích thước pixel
        img_path = os.path.join(img_folder, "img.png")
        img.save(img_path)

        # Xử lý trước hình ảnh để dự đoán
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
        _, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
        roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

        # Làm phẳng hình ảnh thành mảng pixel 1D
        X = [1 if pixel > 100 else 0 for row in roi for pixel in row]

        #tiến hành dự đoán chữ số

        prediction = model.predict([X])[0]

        # Display the predicted result on the image
        cv2.putText(im, f"Prediction: {prediction}", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Prediction", im)

        # Ngắt vòng lặp khi nhấn phím 'Enter' (mã phím là 13)
        if cv2.waitKey(1) == 13:
            break

    cv2.destroyAllWindows()

btn = tk.Button(canvas2, text="livepaint", fg="white", bg="green", command=livepaint)
btn.place(x=40, y=0)



canvas3 = Canvas(window,width=280,height=530,bg='green')
canvas3.place(x=500,y=120)

# (exit_button = tk.Button(window, text="Thoát", fg="white", bg="red", font=('Times New Roman', 15), command=window.destroy)
# ( exit_button.place(x=700, y=70)
# ben tren la giao dien exit neu ko muon nhan vao anh de thoat









window.geometry("800x680")
window.mainloop()