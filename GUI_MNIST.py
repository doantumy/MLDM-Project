# Ref: https://github.com/salvacorts/Keras-MNIST-Paint/
# Ref: https://stackoverflow.com/questions/9886274/how-can-i-convert-canvas-content-to-an-image

import pickle
import time
import warnings
from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from PIL import ImageGrab
from keras import backend as K
from keras.models import load_model

import freeman as fm
import process_utils as pu

warnings.filterwarnings('ignore')

class Paint(object):

    DEFAULT_PEN_SIZE = 10.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()
        self.root.title("Digits Predictor")
        #self.root.resizable(0,0)
        self.root.geometry("641x500")

        im = Image.open("./gui/MNIST_new.png")
        photo = ImageTk.PhotoImage(im)
        self.cv = Canvas(self.root,bg='white', width=500, height=500, highlightthickness=1, highlightbackground="black")
        self.cv.place(x=0, y=0, relwidth=1, relheight=1)
        self.cv.create_image(0,0, image=photo, anchor='nw')

        self.c = Canvas(self.root, bg='white', width=130, height=130, highlightthickness=1, highlightbackground="#aca9a9")
        self.c.place(x=70, y=125)

        blkimg = ImageTk.PhotoImage(Image.open("./data/temp/blank.png"))
        self.cplot = Label(self.root, height=150,width=150, image=blkimg)
        self.cplot.place(x=460, y=278)

        self.predictlbl = Label(self.root, text="PREDICT", bg="#ed4e5b",fg="white",cursor="hand2")
        self.predictlbl.place(x=42, y=456)
        self.predictlbl.config(font=("Helvetica",14,"bold"))
        self.predictlbl.bind("<Button-1>", self.predict)

        self.resetlbl = Label(self.root, text="RESET", bg="#ed4e5b", fg="white",cursor="hand2")
        self.resetlbl.place(x=200, y=456)
        self.resetlbl.config(font=("Helvetica", 14, "bold"))
        self.resetlbl.bind("<Button-1>", self.user_eraser)

        self.freqlbl = Label(self.root, text="FREQUENT SEQUENCE", bg="#ed4e5b", fg="white", cursor="hand2")
        self.freqlbl.place(x=311, y=456)
        self.freqlbl.config(font=("Helvetica", 14, "bold"))
        self.freqlbl.bind("<Button-1>", self.frequent_seq)

        self.predictionval = Label(self.root, text="-", justify="left")
        self.predictionval.place(x=400, y=105)
        self.predictionval.config(font=("Arial",24, "bold"))

        self.time_lbl = Label(self.root, text="-", justify="left")
        self.time_lbl.place(x=400, y=144)

        self.seq_lbl = Label(self.root, text="-", justify="left", wraplength=100)
        self.seq_lbl.place(x=290, y=275)
        self.seq_lbl.config(font=("Courier", 12))

        self.freemanval = Label(self.root, justify="left",text="-",wraplength=250)
        self.freemanval.place(x=290, y=190)
        self.freemanval.config(font=("Courier", 12))

        self.v = IntVar()
        self.r1 = Radiobutton(self.root, text="", variable=self.v, value=0)
        self.r2 = Radiobutton(self.root, text="", variable=self.v, value=1)
        self.r3 = Radiobutton(self.root, text="", variable=self.v, value=2)

        self.r1.place(x=30, y=342)
        self.r2.place(x=30, y=369)
        self.r3.place(x=30, y=397)
        self.r1.config(font=("Arial",15))
        self.r2.config(font=("Arial", 15))
        self.r3.config(font=("Arial", 15))

        self.cnnknn()
        self.editdist()
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.kstrip=10
        self.nn = 5
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        with open('./freq_seq/freq_sequence_all.pickle', 'rb') as f:
            self.freq_sequence_all_load = pickle.load(f)

    def editdist(self):
        self.freeman_X_resampled = np.load("./data/freeman_X_resampled.npy")
        self.y_resampled = np.load('./data/y_resampled_auto_5.npy')
        self.centroids = np.load("./data/centroids.npy")
        self.cluster_lbl = np.load("./data/cluster_lbl.npy")

    def cnnknn(self):
        # Load classifier for CNN+KNN
        with open('./data/best_classifier.pkl', 'rb') as f:
            self.best_classifier_loaded = pickle.load(f)
        new_model_path = "./model/cnn_new_data.h5"
        self.new_loaded_model = load_model(new_model_path)
        self.loaded_feature_layer = K.function([self.new_loaded_model.layers[0].input, K.learning_phase()],
                                          [self.new_loaded_model.layers[11].output])

    def frequent_seq(self, event):
        fq_test = self.freq_sequence_all_load[self.pred_digit]
        print("Orig freq seq: ", fq_test)
        img = pu.process_image("./data/temp/tmp.png")
        test_freeman, test_boundaries = fm.freeman_chain_code(np.float32(img), 'normal')
        test_fm_code = ''.join(test_freeman)
        index, max_sim, sub_boundaries, max_sim_fm = pu.find_subseq_match(test_fm_code,
                                                                          fq_test,
                                                                          test_boundaries)
        self.seq_lbl.config(text=max_sim_fm)
        print(index, max_sim, sub_boundaries, max_sim_fm)
        self.plot_boundaries(img, sub_boundaries)
        blkimg = ImageTk.PhotoImage(Image.open("./data/temp/freq.png"))
        self.cplot.configure(image=blkimg)
        self.cplot.image=blkimg

    def user_eraser(self,event):
        print("Reset")
        self.freemanval.config(text="-")
        self.predictionval.config(text="-")
        self.time_lbl.config(text="-")
        self.seq_lbl.config(text="-")
        blkimg = ImageTk.PhotoImage(Image.open("./data/temp/blank.png"))
        self.cplot.configure(image=blkimg)
        self.c.delete("all")

    def paint(self, event):
        self.line_width = self.DEFAULT_PEN_SIZE
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def predict(self,event):
        print("Predict")
        self.freemanval.config(text="-")
        self.predictionval.config(text="-")
        self.time_lbl.config(text="-")
        self.save(self.c)
        select_val = self.v.get()
        img = pu.process_image("./data/temp/tmp.png")

        if select_val == 2:
            tic = time.time()
            demo_img_norm = pu.normalize_data(img)
            demo_img_reshape = demo_img_norm.reshape(-1, 28, 28, 1)
            demo_input = self.loaded_feature_layer([demo_img_reshape, 0])[0]
            y_test_knn = self.best_classifier_loaded.predict(demo_input)
            toc = time.time()
            self.pred_digit = y_test_knn[0]
            self.time_lbl.config(text="{0:.5f} sec".format(toc-tic))
            self.predictionval.config(text=str(y_test_knn[0]))
            self.freemanval.config(text="-")
        else:
            test_freeman, _ = fm.freeman_chain_code(np.float32(img), 'normal')
            test_fm_code = ''.join(test_freeman)
            print("fm: ", test_fm_code)
            if select_val == 0:
                self.freemanval.config(text=test_fm_code)
                pred, cal_time = pu.knn(test_fm_code, self.freeman_X_resampled, self.y_resampled, self.kstrip, self.nn)
                self.pred_digit = pred
                self.predictionval.config(text=str(pred))
                self.time_lbl.config(text="{0:.5f} sec".format(cal_time))
            elif select_val == 1:
                self.freemanval.config(text=test_fm_code)
                pred, cal_time = pu.kmeans_knn(img, self.centroids, self.cluster_lbl, self.freeman_X_resampled, self.y_resampled, self.kstrip, self.nn)
                self.pred_digit = pred
                self.predictionval.config(text=str(pred))
                self.time_lbl.config(text="{0:.5f} sec".format(cal_time))


    def save(self,widget):
        x=self.root.winfo_rootx()+widget.winfo_x()
        y=self.root.winfo_rooty()+widget.winfo_y()
        x1=x+widget.winfo_width()
        y1=y+widget.winfo_height()
        ImageGrab.grab().crop((x,y,x1,y1)).save("./data/temp/tmp.png")


    def plot_boundaries(self, image_matrix, boundaries, color='yellow'):
        plt.axis('off')
        plt.gcf().clear()
        plt.imshow(image_matrix, cmap="Greys")
        # Plot the countours
        plt.plot([pixel[1] for pixel in boundaries],
                 [pixel[0] for pixel in boundaries],
                 color=color, linewidth=4)
        plt.savefig("./data/temp/freq.png", dpi=40, pad_inches = 0)
       # plt.show()

if __name__ == '__main__':
    ge = Paint()
