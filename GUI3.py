import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QComboBox, QSlider, QSpinBox,
                             QDoubleSpinBox, QGroupBox, QSplitter)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理系统")
        self.setGeometry(100, 100, 1200, 800)

        self.initUI()
        self.image = None
        self.processed_image = None

    def initUI(self):
        # 主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 分割左右面板
        splitter = QSplitter(Qt.Horizontal)

        # 左侧面板 - 图像显示
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 原图像显示
        self.original_image_label = QLabel("原图像")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 300)
        self.original_image_label.setStyleSheet("border: 1px solid black;")
        left_layout.addWidget(self.original_image_label)

        # 处理后的图像显示
        self.processed_image_label = QLabel("处理后图像")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setMinimumSize(400, 300)
        self.processed_image_label.setStyleSheet("border: 1px solid black;")
        left_layout.addWidget(self.processed_image_label)

        # 右侧面板 - 控制区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QHBoxLayout(file_group)

        self.load_button = QPushButton("加载图像")
        self.load_button.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_button)

        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_image)
        file_layout.addWidget(self.save_button)

        right_layout.addWidget(file_group)

        # 直方图操作
        hist_group = QGroupBox("直方图操作")
        hist_layout = QVBoxLayout(hist_group)

        self.hist_button = QPushButton("显示直方图")
        self.hist_button.clicked.connect(self.show_histogram)
        hist_layout.addWidget(self.hist_button)

        self.equalize_button = QPushButton("直方图均衡化")
        self.equalize_button.clicked.connect(self.equalize_histogram)
        hist_layout.addWidget(self.equalize_button)

        # 直方图画布
        self.hist_figure = Figure(figsize=(5, 3))
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_canvas.setMinimumSize(400, 250)
        hist_layout.addWidget(self.hist_canvas)

        right_layout.addWidget(hist_group)

        # 对比度增强
        contrast_group = QGroupBox("对比度增强 (伽马校正)")
        contrast_layout = QVBoxLayout(contrast_group)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(1)
        self.gamma_slider.setMaximum(300)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self.apply_gamma_correction)

        self.gamma_label = QLabel("伽马值: 1.0")
        contrast_layout.addWidget(self.gamma_label)
        contrast_layout.addWidget(self.gamma_slider)

        right_layout.addWidget(contrast_group)

        # 几何变换
        geom_group = QGroupBox("几何变换")
        geom_layout = QVBoxLayout(geom_group)

        # 平移
        trans_layout = QHBoxLayout()
        trans_layout.addWidget(QLabel("平移 X:"))
        self.trans_x = QSpinBox()
        self.trans_x.setRange(-500, 500)
        self.trans_x.setValue(0)
        trans_layout.addWidget(self.trans_x)

        trans_layout.addWidget(QLabel("平移 Y:"))
        self.trans_y = QSpinBox()
        self.trans_y.setRange(-500, 500)
        self.trans_y.setValue(0)
        trans_layout.addWidget(self.trans_y)

        self.trans_button = QPushButton("应用平移")
        self.trans_button.clicked.connect(self.apply_translation)
        trans_layout.addWidget(self.trans_button)

        geom_layout.addLayout(trans_layout)

        # 旋转
        rotate_layout = QHBoxLayout()
        rotate_layout.addWidget(QLabel("旋转角度:"))
        self.rotate_angle = QSpinBox()
        self.rotate_angle.setRange(-360, 360)
        self.rotate_angle.setValue(0)
        rotate_layout.addWidget(self.rotate_angle)

        self.rotate_button = QPushButton("应用旋转")
        self.rotate_button.clicked.connect(self.apply_rotation)
        rotate_layout.addWidget(self.rotate_button)

        geom_layout.addLayout(rotate_layout)
        right_layout.addWidget(geom_group)

        # 噪声和滤波
        noise_group = QGroupBox("噪声和滤波")
        noise_layout = QVBoxLayout(noise_group)

        # 高斯噪声
        gauss_noise_layout = QHBoxLayout()
        gauss_noise_layout.addWidget(QLabel("均值:"))
        self.noise_mean = QDoubleSpinBox()
        self.noise_mean.setRange(0, 100)
        self.noise_mean.setSingleStep(0.1)
        self.noise_mean.setValue(0)
        gauss_noise_layout.addWidget(self.noise_mean)

        gauss_noise_layout.addWidget(QLabel("标准差:"))
        self.noise_sigma = QDoubleSpinBox()
        self.noise_sigma.setRange(0, 100)
        self.noise_sigma.setSingleStep(0.1)
        self.noise_sigma.setValue(10)
        gauss_noise_layout.addWidget(self.noise_sigma)

        self.add_noise_button = QPushButton("添加高斯噪声")
        self.add_noise_button.clicked.connect(self.add_gaussian_noise)
        gauss_noise_layout.addWidget(self.add_noise_button)

        noise_layout.addLayout(gauss_noise_layout)

        # 泊松噪声
        poisson_layout = QHBoxLayout()
        poisson_layout.addWidget(QLabel("Lambda:"))
        self.poisson_lambda = QDoubleSpinBox()
        self.poisson_lambda.setRange(0.1, 100.0)
        self.poisson_lambda.setSingleStep(0.1)
        self.poisson_lambda.setValue(10.0)
        poisson_layout.addWidget(self.poisson_lambda)

        self.poisson_button = QPushButton("添加泊松噪声")
        self.poisson_button.clicked.connect(self.add_poisson_noise)
        poisson_layout.addWidget(self.poisson_button)

        noise_layout.addLayout(poisson_layout)

        # 椒盐噪声
        salt_pepper_layout = QHBoxLayout()
        salt_pepper_layout.addWidget(QLabel("噪声比例:"))
        self.salt_pepper_amount = QDoubleSpinBox()
        self.salt_pepper_amount.setRange(0.0, 1.0)
        self.salt_pepper_amount.setSingleStep(0.01)
        self.salt_pepper_amount.setValue(0.05)
        salt_pepper_layout.addWidget(self.salt_pepper_amount)

        salt_pepper_layout.addWidget(QLabel("盐/椒比例:"))
        self.salt_vs_pepper = QDoubleSpinBox()
        self.salt_vs_pepper.setRange(0.0, 1.0)
        self.salt_vs_pepper.setSingleStep(0.1)
        self.salt_vs_pepper.setValue(0.5)
        salt_pepper_layout.addWidget(self.salt_vs_pepper)

        self.salt_pepper_button = QPushButton("添加椒盐噪声")
        self.salt_pepper_button.clicked.connect(self.add_salt_pepper_noise)
        salt_pepper_layout.addWidget(self.salt_pepper_button)

        noise_layout.addLayout(salt_pepper_layout)

        # 空间滤波
        spatial_layout = QHBoxLayout()
        spatial_layout.addWidget(QLabel("滤波器大小:"))
        self.spatial_size = QSpinBox()
        self.spatial_size.setRange(1, 31)
        self.spatial_size.setSingleStep(2)
        self.spatial_size.setValue(5)
        spatial_layout.addWidget(self.spatial_size)

        spatial_layout.addWidget(QLabel("Sigma:"))
        self.spatial_sigma = QDoubleSpinBox()
        self.spatial_sigma.setRange(0.1, 10.0)
        self.spatial_sigma.setSingleStep(0.1)
        self.spatial_sigma.setValue(1.5)
        spatial_layout.addWidget(self.spatial_sigma)

        self.spatial_button = QPushButton("高斯平滑(空域)")
        self.spatial_button.clicked.connect(self.apply_spatial_filter)
        spatial_layout.addWidget(self.spatial_button)

        noise_layout.addLayout(spatial_layout)

        # 频域滤波
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("截止频率:"))
        self.cutoff_freq = QSpinBox()
        self.cutoff_freq.setRange(1, 100)
        self.cutoff_freq.setValue(30)
        freq_layout.addWidget(self.cutoff_freq)

        self.freq_button = QPushButton("高斯低通(频域)")
        self.freq_button.clicked.connect(self.apply_frequency_filter)
        freq_layout.addWidget(self.freq_button)

        noise_layout.addLayout(freq_layout)

        right_layout.addWidget(noise_group)

        # 将左右面板添加到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "图像文件 (*.png *.jpg *.bmp *.tif)")
        if file_name:
            self.image = cv2.imread(file_name)
            if self.image is not None:
                self.display_image(self.image, self.original_image_label)
                self.processed_image = self.image.copy()
                self.display_image(self.processed_image, self.processed_image_label)

    def save_image(self):
        if self.processed_image is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "保存图像", "",
                                                       "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*)")
            if file_name:
                cv2.imwrite(file_name, self.processed_image)

    def display_image(self, image, label):
        if len(image.shape) == 2:  # 灰度图像
            q_img = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Grayscale8)
        else:  # 彩色图像
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def show_histogram(self):
        if self.image is None:
            return

        # 转换为灰度图像
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        # 绘制直方图
        self.hist_figure.clear()
        ax = self.hist_figure.add_subplot(111)
        ax.plot(hist, color='black')
        ax.set_title('Gray Histogram', fontsize=10, pad=10)
        ax.set_xlabel('Gray Level', fontsize=8, labelpad=5)
        ax.set_ylabel('Pixel Count', fontsize=8, labelpad=5)
        ax.set_xlim([0, 255])
        self.hist_figure.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.85)
        ax.tick_params(axis='both', which='major', labelsize=8)
        self.hist_canvas.draw()

    def equalize_histogram(self):
        if self.image is None:
            return

        # 转换为灰度图像
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        # 直方图均衡化
        equalized = cv2.equalizeHist(gray)

        # 显示处理后的图像
        self.processed_image = equalized
        self.display_image(self.processed_image, self.processed_image_label)

        # 显示均衡化后的直方图
        hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])
        self.hist_figure.clear()
        ax = self.hist_figure.add_subplot(111)
        ax.plot(hist, color='black')
        ax.set_title('Gray Histogram', fontsize=10, pad=10)
        ax.set_xlabel('Gray Level', fontsize=8, labelpad=5)
        ax.set_ylabel('Pixel Count', fontsize=8, labelpad=5)
        ax.set_xlim([0, 255])
        self.hist_figure.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.85)
        ax.tick_params(axis='both', which='major', labelsize=8)
        self.hist_canvas.draw()

    def apply_gamma_correction(self):
        if self.image is None:
            return

        gamma = self.gamma_slider.value() / 100.0
        self.gamma_label.setText(f"伽马值: {gamma:.2f}")

        # 转换为灰度图像
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        # 归一化
        gray_norm = gray / 255.0

        # 应用伽马校正
        corrected = np.power(gray_norm, gamma)
        corrected = (corrected * 255).astype(np.uint8)

        # 显示处理后的图像
        self.processed_image = corrected
        self.display_image(self.processed_image, self.processed_image_label)

    def apply_translation(self):
        if self.image is None:
            return

        tx = self.trans_x.value()
        ty = self.trans_y.value()

        rows, cols = self.image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(self.image, M, (cols, rows))

        self.processed_image = translated
        self.display_image(self.processed_image, self.processed_image_label)

    def apply_rotation(self):
        if self.image is None:
            return

        angle = self.rotate_angle.value()
        rows, cols = self.image.shape[:2]

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(self.image, M, (cols, rows))

        self.processed_image = rotated
        self.display_image(self.processed_image, self.processed_image_label)

    def add_gaussian_noise(self):
        if self.image is None:
            return

        mean = self.noise_mean.value()
        sigma = self.noise_sigma.value()

        # 生成高斯噪声
        if len(self.image.shape) == 3:
            h, w, c = self.image.shape
            noise = np.random.normal(mean, sigma, (h, w, c)).astype(np.uint8)
            noisy = cv2.add(self.image, noise)
        else:
            h, w = self.image.shape
            noise = np.random.normal(mean, sigma, (h, w)).astype(np.uint8)
            noisy = cv2.add(self.image, noise)

        self.processed_image = noisy
        self.display_image(self.processed_image, self.processed_image_label)

    def add_poisson_noise(self):
        if self.image is None:
            return

        lam = self.poisson_lambda.value()

        # 生成泊松噪声
        if len(self.image.shape) == 3:
            # 彩色图像
            noisy = np.zeros_like(self.image, dtype=np.float32)
            for i in range(3):
                channel = self.image[:, :, i].astype(np.float32)
                noise = np.random.poisson(lam * channel / 255.0) * (255.0 / lam)
                noisy[:, :, i] = np.clip(noise, 0, 255)
            noisy = noisy.astype(np.uint8)
        else:
            # 灰度图像
            noise = np.random.poisson(lam * self.image.astype(np.float32) / 255.0) * (255.0 / lam)
            noisy = np.clip(noise, 0, 255).astype(np.uint8)

        self.processed_image = noisy
        self.display_image(self.processed_image, self.processed_image_label)

    def add_salt_pepper_noise(self):
        if self.image is None:
            return

        amount = self.salt_pepper_amount.value()
        s_vs_p = self.salt_vs_pepper.value()

        # 生成椒盐噪声
        noisy = np.copy(self.image)

        # 盐噪声
        num_salt = np.ceil(amount * self.image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.image.shape[:2]]
        if len(self.image.shape) == 3:
            for i in range(3):
                noisy[coords[0], coords[1], i] = 255
        else:
            noisy[coords[0], coords[1]] = 255

        # 椒噪声
        num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.image.shape[:2]]
        if len(self.image.shape) == 3:
            for i in range(3):
                noisy[coords[0], coords[1], i] = 0
        else:
            noisy[coords[0], coords[1]] = 0

        self.processed_image = noisy
        self.display_image(self.processed_image, self.processed_image_label)

    def apply_spatial_filter(self):
        if self.image is None:
            return

        ksize = self.spatial_size.value()
        sigma = self.spatial_sigma.value()

        # 确保ksize是奇数
        if ksize % 2 == 0:
            ksize += 1

        filtered = cv2.GaussianBlur(self.processed_image, (ksize, ksize), sigma)

        self.processed_image = filtered
        self.display_image(self.processed_image, self.processed_image_label)

    def apply_frequency_filter(self):
        if self.image is None:
            return

        cutoff = self.cutoff_freq.value()

        # 转换为灰度图像
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image

        # 傅里叶变换
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # 创建高斯滤波器
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.float32)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                mask[i, j] = np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))

        # 应用滤波器
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_REAL_OUTPUT)
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self.processed_image = img_back
        self.display_image(self.processed_image, self.processed_image_label)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())
