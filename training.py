import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import time

MODEL_PATH = 'mnist_cnn_model.h5'
RESULTS_DIR = 'training_results'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("Завантаження даних MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Нормалізація зображень
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Зміна форми для вхідних даних CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot кодування міток
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Форма навчальних даних: {X_train.shape}")
print(f"Форма тестових даних: {X_test.shape}")

def create_cnn_model():
    model = models.Sequential()

    # Згорткові шари
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Повнозв'язні шари
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Функція для навчання моделі з різними оптимізаторами та функціями втрат
def train_and_evaluate(optimizer_name, loss_function, epochs=10, batch_size=64):
    print(f"\nНавчання моделі з оптимізатором {optimizer_name} та функцією втрат {loss_function}")

    model = create_cnn_model()

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    else:
        raise ValueError(f"Невідомий оптимізатор: {optimizer_name}")

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])

    # Навчання моделі
    start_time = time.time()
    history = model.fit(X_train, y_train_cat,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test_cat),
                        verbose=1)

    training_time = time.time() - start_time
    print(f"Час навчання: {training_time:.2f} секунд")

    # Оцінка моделі
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Тестова точність: {test_acc:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Обчислення матриці помилок
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    results = {
        'model': model,
        'history': history,
        'test_accuracy': test_acc,
        'confusion_matrix': conf_matrix,
        'training_time': training_time,
        'y_true': y_test,
        'y_pred': y_pred_classes
    }

    save_training_plots(history, conf_matrix, optimizer_name, loss_function)

    return results

def save_training_plots(history, conf_matrix, optimizer_name, loss_function):
    config_name = f"{optimizer_name}_{loss_function}"

    # Графік точності
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Навчальна вибірка')
    plt.plot(history.history['val_accuracy'], label='Тестова вибірка')
    plt.title(f'Точність моделі ({config_name})')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.legend()

    # Графік втрат
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Навчальна вибірка')
    plt.plot(history.history['val_loss'], label='Тестова вибірка')
    plt.title(f'Втрати моделі ({config_name})')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_history_{config_name}.png")
    plt.close()

    # Матриця помилок
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Матриця помилок ({config_name})')
    plt.xlabel('Передбачені мітки')
    plt.ylabel('Істинні мітки')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_{config_name}.png")
    plt.close()

# Навчання моделей з різними конфігураціями
configurations = [
    ('adam', 'categorical_crossentropy'),
    ('sgd', 'categorical_crossentropy'),
    ('rmsprop', 'categorical_crossentropy'),
    ('adam', 'mean_squared_error')
]

results = {}
best_accuracy = 0
best_config = None

# Навчання та оцінка моделі для кожної конфігурації
for optimizer, loss_function in configurations:
    config_name = f"{optimizer}_{loss_function}"
    print(f"\n{'='*50}")
    print(f"Конфігурація: {config_name}")
    print(f"{'='*50}")

    # Навчання та оцінка моделі
    result = train_and_evaluate(optimizer, loss_function, epochs=5, batch_size=64)
    results[config_name] = result

    # Збереження найкращої моделі
    if result['test_accuracy'] > best_accuracy:
        best_accuracy = result['test_accuracy']
        best_config = config_name
        result['model'].save(MODEL_PATH)

# Виведення результатів навчання
print("\nРезультати навчання:")
print(f"{'='*50}")
for config, result in results.items():
    print(f"Конфігурація: {config}")
    print(f"Точність: {result['test_accuracy']:.4f}")
    print(f"Час навчання: {result['training_time']:.2f} секунд")

    # Виведення звіту класифікації
    print("\nЗвіт класифікації:")
    print(classification_report(result['y_true'], result['y_pred'], digits=4))
    print(f"{'-'*50}")

print(f"\nНайкраща модель: {best_config} з точністю {best_accuracy:.4f}")
print(f"Модель збережена у файл: {MODEL_PATH}")