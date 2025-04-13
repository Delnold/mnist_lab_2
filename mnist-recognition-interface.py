import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys


class DigitRecognizer:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Розпізнавання рукописних цифр MNIST")
        self.root.geometry("1000x620")
        self.root.configure(bg='#f0f0f0')

        self.setup_styles()

        try:
            self.model = load_model(model_path)
            print(f"Модель завантажена з файлу: {model_path}")
        except Exception as e:
            messagebox.showerror("Помилка завантаження моделі",
                                 f"Не вдалося завантажити модель з файлу {model_path}.\n"
                                 f"Помилка: {str(e)}")
            sys.exit(1)

        self.setup_ui()

    def setup_styles(self):
        """Налаштовує стилі для покращення візуального вигляду"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TLabel', font=('Helvetica', 11), background='#f0f0f0')
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'), background='#f0f0f0')

        style.configure('TButton', font=('Helvetica', 11), padding=5)
        style.map('TButton',
                  background=[('active', '#4a7abc')],
                  foreground=[('active', 'white')])

        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', font=('Helvetica', 11, 'bold'), background='#f0f0f0')

    def setup_ui(self):
        """Налаштовує компоненти інтерфейсу"""
        main_frame = ttk.Frame(self.root, padding=10, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Розпізнавання рукописних цифр MNIST",
                                style='Title.TLabel')
        title_label.pack(pady=10)

        params_frame = ttk.Frame(main_frame, style='TFrame')
        params_frame.pack(fill=tk.X, pady=5)

        thickness_label = ttk.Label(params_frame, text="Товщина лінії:", style='TLabel')
        thickness_label.pack(side=tk.LEFT, padx=(0, 5))

        self.thickness_var = tk.IntVar(value=2)
        thickness_scale = ttk.Scale(params_frame, from_=1, to=5,
                                    orient=tk.HORIZONTAL, length=150,
                                    variable=self.thickness_var)
        thickness_scale.pack(side=tk.LEFT, padx=(0, 20))

        size_label = ttk.Label(params_frame, text="Розмір:", style='TLabel')
        size_label.pack(side=tk.LEFT, padx=(0, 5))

        self.size_var = tk.StringVar(value="28x28")
        size_combo = ttk.Combobox(params_frame, textvariable=self.size_var,
                                  values=["28x28"], state="readonly", width=10)
        size_combo.pack(side=tk.LEFT, padx=(0, 20))

        content_frame = ttk.Frame(main_frame, style='TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        drawing_frame = ttk.LabelFrame(content_frame, text="Полотно для малювання", style='TLabelframe')
        drawing_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)

        canvas_display_size = 280  # Збільшено для зручності відображення
        self.canvas_size = 28  # Реальний розмір для моделі

        canvas_frame = ttk.Frame(drawing_frame, style='TFrame')
        canvas_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame,
                                width=canvas_display_size,
                                height=canvas_display_size,
                                bg='black',
                                cursor="pencil",
                                highlightbackground='#4a7abc',
                                highlightthickness=2)
        self.canvas.pack(padx=10, pady=10)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.draw_grid()

        self.setup_mouse_events()

        button_frame = ttk.Frame(drawing_frame, style='TFrame')
        button_frame.pack(fill=tk.X, pady=5)

        self.predict_button = ttk.Button(button_frame, text="Розпізнати",
                                         command=self.predict_digit, style='TButton')
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ttk.Button(button_frame, text="Очистити",
                                       command=self.clear_canvas, style='TButton')
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(button_frame, text="Зберегти зображення",
                                      command=self.save_image, style='TButton')
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(button_frame, text="Завантажити зображення",
                                      command=self.load_image, style='TButton')
        self.load_button.pack(side=tk.LEFT, padx=5)

        result_frame = ttk.LabelFrame(content_frame, text="Результат розпізнавання", style='TLabelframe')
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

        self.prediction_label = ttk.Label(result_frame,
                                          text="Намалюйте цифру і натисніть 'Розпізнати'",
                                          font=('Helvetica', 12), style='TLabel')
        self.prediction_label.pack(pady=10)

        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas_prediction = FigureCanvasTkAgg(self.fig, master=result_frame)
        self.canvas_prediction.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.update_prediction_graph([0] * 10)

        self.status_var = tk.StringVar(value="Готовий до роботи")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W, style='TLabel')
        status_bar.pack(fill=tk.X, pady=(5, 0))

    def draw_grid(self):
        """Відображає сітку 28x28 на полотні"""
        grid_visible = False  # Змініть на True, щоб відобразити сітку

        if grid_visible:
            canvas_width = self.canvas.winfo_reqwidth()
            canvas_height = self.canvas.winfo_reqheight()

            cell_width = canvas_width / self.canvas_size
            cell_height = canvas_height / self.canvas_size

            for i in range(self.canvas_size + 1):
                x = i * cell_width
                self.canvas.create_line(x, 0, x, canvas_height, fill="#333333", width=0.5)

            for i in range(self.canvas_size + 1):
                y = i * cell_height
                self.canvas.create_line(0, y, canvas_width, y, fill="#333333", width=0.5)

    def setup_mouse_events(self):
        """Налаштовує обробники подій миші для малювання"""
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", lambda event: None)

    def start_drawing(self, event):
        """Обробник події натискання кнопки миші"""
        self.last_x, self.last_y = event.x, event.y

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        x_rel = event.x / canvas_width
        y_rel = event.y / canvas_height

        img_x = int(x_rel * self.canvas_size)
        img_y = int(y_rel * self.canvas_size)

        img_x = max(0, min(img_x, self.canvas_size - 1))
        img_y = max(0, min(img_y, self.canvas_size - 1))

        thickness = self.thickness_var.get()
        self.draw.ellipse([(img_x - thickness / 2, img_y - thickness / 2),
                           (img_x + thickness / 2, img_y + thickness / 2)], fill=255)

        self.last_img_x, self.last_img_y = img_x, img_y

        self.update_canvas_display()

    def draw_line(self, event):
        """Обробник події руху миші з натиснутою кнопкою"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        x_rel = event.x / canvas_width
        y_rel = event.y / canvas_height

        img_x = int(x_rel * self.canvas_size)
        img_y = int(y_rel * self.canvas_size)

        img_x = max(0, min(img_x, self.canvas_size - 1))
        img_y = max(0, min(img_y, self.canvas_size - 1))

        thickness = self.thickness_var.get()
        self.draw.line([(self.last_img_x, self.last_img_y), (img_x, img_y)],
                       fill=255, width=thickness)

        self.last_img_x, self.last_img_y = img_x, img_y

        self.update_canvas_display()

    def update_canvas_display(self):
        """Оновлює відображення малювання на полотні"""
        display_img = self.image.resize((280, 280), Image.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(display_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.draw_grid()

    def clear_canvas(self):
        """Очищає полотно для малювання"""
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.update_canvas_display()

        self.prediction_label.config(text="Намалюйте цифру і натисніть 'Розпізнати'")
        self.update_prediction_graph([0] * 10)
        self.status_var.set("Полотно очищено")

    def save_image(self):
        """Зберігає намальоване зображення"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG файли", "*.png"), ("Всі файли", "*.*")]
        )
        if file_path:
            self.image.save(file_path)

            large_path = os.path.splitext(file_path)[0] + "_large.png"
            large_image = self.image.resize((280, 280), Image.LANCZOS)
            large_image.save(large_path)

            self.status_var.set(f"Зображення збережено як: {os.path.basename(file_path)}")

    def load_image(self):
        """Завантажує зображення для розпізнавання"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Файли зображень", "*.png;*.jpg;*.jpeg;*.bmp"), ("Всі файли", "*.*")]
        )
        if file_path:
            try:
                loaded_image = Image.open(file_path).convert('L')

                resized_image = loaded_image.resize((self.canvas_size, self.canvas_size), Image.LANCZOS)

                self.image = resized_image
                self.draw = ImageDraw.Draw(self.image)

                self.update_canvas_display()

                self.predict_digit()

                self.status_var.set(f"Завантажено: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Помилка завантаження", str(e))
                self.status_var.set("Помилка завантаження зображення")

    def predict_digit(self):
        """Розпізнає намальовану цифру"""
        img_array = np.array(self.image)
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = self.model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class] * 100

        self.prediction_label.config(
            text=f"Розпізнана цифра: {predicted_class} (впевненість: {confidence:.2f}%)")

        self.update_prediction_graph(prediction)

        self.status_var.set(f"Розпізнано: цифра {predicted_class} з впевненістю {confidence:.2f}%")

    def update_prediction_graph(self, predictions):
        """Оновлює графік з ймовірностями розпізнавання"""
        self.ax.clear()
        bars = self.ax.bar(range(10), predictions, color='skyblue')

        if max(predictions) > 0:
            max_idx = np.argmax(predictions)
            bars[max_idx].set_color('#4a7abc')

        self.ax.set_xticks(range(10))
        self.ax.set_ylim([0, 1])
        self.ax.set_xlabel('Цифра')
        self.ax.set_ylabel('Ймовірність')
        self.ax.set_title('Розподіл ймовірностей розпізнавання')

        # Додавання сітки для кращої читабельності
        self.ax.grid(True, alpha=0.3)

        # Налаштування фону
        self.ax.set_facecolor('#f8f8f8')

        self.fig.tight_layout()
        self.canvas_prediction.draw()


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'mnist_cnn_model.h5'

    if not os.path.exists(model_path):
        print(f"Помилка: Файл моделі '{model_path}' не знайдено.")
        print("Використання: python mnist_interface.py [шлях_до_моделі]")
        sys.exit(1)

    root = tk.Tk()
    app = DigitRecognizer(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()