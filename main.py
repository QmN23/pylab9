import numpy as np
from scipy.stats import multivariate_normal

# Task 1
# Сохраняем матрицу в файл
matrix_text = """
3,4,17,-3
5,11,-1,6
0,2,-5,8"""

with open("matrix.txt", "w") as file:
    file.write(matrix_text)

# Читаем матрицу из файла
matrix = np.genfromtxt("matrix.txt", delimiter=',')

# Находим сумму всех элементов, максимальный и минимальный элемент
matrix_sum = np.sum(matrix)
matrix_max = np.max(matrix)
matrix_min = np.min(matrix)

print("Сумма всех элементов матрицы:", matrix_sum)
print("Максимальный элемент матрицы:", matrix_max)
print("Минимальный элемент матрицы:", matrix_min)


# Task2

def run_length_encoding(x):
    unique_values, counts = np.unique(x, return_counts=True)
    return unique_values, counts


x = np.array([2, 2, 2, 3, 3, 3, 5])

values, counts = run_length_encoding(x)
print(values, counts)

# Task 3
# Генерируем массив случайных чисел нормального распределения
array = np.random.normal(size=(10, 4))

# Находим минимум, максимум, среднее и стандартное отклонение
min_value = array.min()
max_value = array.max()
mean_value = array.mean()
std = array.std()

# Сохраняем первые 5 строк в отдельную переменную
first_rows = array[:5]

print("Минимальное значение:", min_value)
print("Максимальное значение:", max_value)
print("Среднее значение:", mean_value)
print("Стандартное отклонение:", std)

# Task 4
x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])

# Ищем индексы нулевых элементов
zero_indices = np.where(x == 0)[0]

valid_indices = zero_indices[zero_indices < len(x) - 1]

# Находим максимальный элемент после нулевого
maxx = np.max(x[valid_indices + 1])

print("Максимальный элемент:", maxx)


# Task 5
def log_density(X, m, C):
    D = X.shape[1]
    # Вычисление определителя матрицы и обратной матрицы
    det_C = np.linalg.det(C)
    inv_C = np.linalg.inv(C)
    # Вычисление логарифма плотности
    log_density = -0.5 * (D * np.log(2 * np.pi) + np.log(det_C) + np.sum((X - m) @ inv_C * (X - m), axis=1))
    return log_density


X = np.random.randn(100, 3)
m = np.mean(X, axis=0)
C = np.cov(X, rowvar=False)
log_density_values = log_density(X, m, C)
scipy_log_density_values = multivariate_normal(m, C).logpdf(X)
print(log_density_values)
print(scipy_log_density_values)

# Task 6

a = np.arange(16).reshape(4,4)
a[[0, 2]] = a[[2, 0]]
print(a)

# Task 7
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Выделяем столбец species
species_column = iris[:, -1]

# Находим уникальные значения и их количество
unique_values, counts = np.unique(species_column, return_counts=True)

print("Уникальные значения:", unique_values)
print("Количество каждого значения:", counts)

# Task 8
x = np.array([0, 1, 2, 0, 0, 4, 0, 6, 9])
nonzero = np.nonzero(x)[0]
print("Индексы ненулевых элементов:", nonzero)
