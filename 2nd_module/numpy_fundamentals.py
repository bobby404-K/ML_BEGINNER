"""
NumPy Fundamentals

A structured walkthrough of core NumPy concepts: creating arrays, inspecting
their attributes, indexing, aggregation, array-creation helpers, broadcasting,
comparisons, reshaping, and the most commonly used mathematical functions
(each with a worked example).
"""

import numpy as np


# ======================================================================
# 1. Creating Arrays
#
# NumPy arrays (`ndarray`) are the core data structure of the library. They can be created from Python lists using `np.array()`.
# ======================================================================

# A 1-D array (vector)
arr_1d = np.array([12, 13, 14, 15])
print(arr_1d, type(arr_1d))


# ======================================================================
# Python lists vs. NumPy arrays
#
# Python's `+` and `*` operators behave very differently on lists compared to arrays:
# - On a list, `+` concatenates and `*` repeats.
# - On a NumPy array, `+` and `*` are element-wise (vectorized) math operations.
# ======================================================================

list_a = [12, 13, 14, 15]
list_b = [5, 6, 7, 8, 9]

concatenated = list_a + list_b   # concatenation, NOT addition
repeated = list_a * 2            # repeats the list twice

print(concatenated, repeated)


array_a = np.array([1, 2, 3, 4, 5])
array_b = np.array([6, 7, 8, 9, 10])

elementwise_sum = array_a + array_b   # element-wise addition
scaled = array_a * 2                  # element-wise multiplication

print(elementwise_sum, scaled)


# ======================================================================
# Note on mixing lists and arrays: when a plain list is combined with a NumPy array using `+`, NumPy converts the list to an array first, so the result is element-wise addition — *not* concatenation. This only works if both are the same length.
# ======================================================================

python_list = [12, 3, 1]
numpy_array = np.array([4, 5, 6])

mixed_result = python_list + numpy_array   # list is coerced to an array
print(mixed_result)


# ======================================================================
# 2. Array Attributes
#
# Renamed from the original `dih` to `matrix_2d` for clarity — the name now reflects what the variable actually holds (a 2-dimensional array).
# ======================================================================

matrix_2d = np.array([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10]])
print(matrix_2d)


print(matrix_2d.ndim)  # number of dimensions (axes)


print(matrix_2d.shape)  # size along each dimension: (rows, columns)


print(matrix_2d.dtype)  # data type of the elements


print(matrix_2d.size)  # total number of elements (rows * columns)


# ======================================================================
# Indexing
# ======================================================================

print(matrix_2d[1])  # the second row (index 1)


# ======================================================================
# Aggregation methods
#
# These summarize all elements of the array.
# ======================================================================

print(matrix_2d.sum())  # sum of all elements


print(matrix_2d.max())  # largest element


print(matrix_2d.mean())  # average of all elements


print(matrix_2d.min())  # smallest element


print(matrix_2d.std())  # standard deviation of all elements


# ======================================================================
# 3. Array-Creation Functions
#
# NumPy provides several convenience functions for building arrays without typing out every element by hand.
# ======================================================================

zeros_list = [0] * 12   # plain Python list of zeros, for comparison
print(zeros_list)


zeros_array = np.zeros(12)   # 1-D array of zeros (float64 by default)
print(zeros_array.dtype, zeros_array)


# A 3-D array of zeros: 7 "layers" of 4x4 grids, stored as integers
zeros_3d = np.zeros((7, 4, 4), dtype=int)
print(zeros_3d.shape, zeros_3d)


# np.full(shape, fill_value) creates an array pre-filled with a constant
filled_array = np.full((4, 7), 6)
print(filled_array)


# np.arange(start, stop, step) -> stop is EXCLUSIVE
range_array = np.arange(2, 8, 1)
print(range_array)


# np.linspace(start, stop, num) -> stop is INCLUSIVE; returns `num` evenly spaced values
evenly_spaced = np.linspace(2, 8, 8)
print(evenly_spaced)


# np.random.random(shape) -> array of random floats in [0.0, 1.0)
random_array = np.random.random((4, 8))
print(random_array.shape)


# ======================================================================
# 4. Finding Extremes: `max`/`min` vs. `argmax`/`argmin`
#
# - `np.max` / `np.min` return the value.
# - `np.argmax` / `np.argmin` return the index at which that value occurs.
# ======================================================================

sample_values = np.array([1, 5, 6, 7, 8, 45, 64, 46])

largest_value = np.max(sample_values)     # value
largest_index = np.argmax(sample_values)  # index of that value

print(largest_value, largest_index)


smallest_value = np.min(sample_values)
smallest_index = np.argmin(sample_values)

print(smallest_value, smallest_index)


# ======================================================================
# 5. Mathematical Functions
#
# General pattern: `np.functionname(operand)` or `np.functionname(op1, op2)`. Every function below is shown on both a single value and a whole array (NumPy applies these element-wise automatically).
# ======================================================================

# ======================================================================
# 5.1 Exponentials & Logarithms
# ======================================================================

print(np.log(10))  # natural log (base e) of 10


print(np.exp(2))  # e raised to the power 2


print(np.sqrt(6))  # square root of 6


perfect_squares = np.array([1, 4, 9, 25, 36, 49, 64, 81, 100])
print(np.sqrt(perfect_squares))  # sqrt applied element-wise to the whole array


print(np.log2(8))  # log base 2


print(np.log10(1000))  # log base 10


print(np.power(2, 5))  # 2 raised to the power 5


print(np.power(np.array([1, 2, 3, 4]), 2))  # element-wise squaring


# ======================================================================
# 5.2 Basic Arithmetic Functions (equivalent to `+ - * /`)
# ======================================================================

base_values = np.array([1, 4, 9, 25, 36, 49, 64, 81, 100])

added        = base_values + 1     # np.add(base_values, 1)
subtracted   = base_values - 3     # np.subtract(base_values, 3)
multiplied   = base_values * 12    # np.multiply(base_values, 12)
divided      = base_values / 7     # np.divide(base_values, 7)

print(added, subtracted, multiplied, divided)


print(np.add(5, 3), np.subtract(5, 3), np.multiply(5, 3), np.divide(5, 3))


# ======================================================================
# 5.3 Trigonometric Functions
# ======================================================================

angle_rad = np.pi / 2   # 90 degrees in radians
print(np.sin(angle_rad), np.cos(angle_rad), np.tan(np.pi / 4))


angles = np.array([0, np.pi / 2, np.pi])
print(np.sin(angles))  # applied element-wise


# ======================================================================
# 5.4 Rounding Functions
# ======================================================================

value = 4.567
print(np.round(value, 2), np.floor(value), np.ceil(value))


decimals = np.array([1.2, 2.7, -3.5, 4.49])
print(np.round(decimals), np.floor(decimals), np.ceil(decimals))


# ======================================================================
# 5.5 Absolute Value & Modulo
# ======================================================================

print(np.abs(-7), np.abs(np.array([-3, -1, 2, -8])))


print(np.mod(10, 3))  # remainder of 10 / 3


# ======================================================================
# 6. Comparison Functions
# ======================================================================

tt = np.array([3, 4, 5, 6])
td = np.array([3, -1, 4, 6])

print(np.equal(tt, td))  # element-wise comparison -> boolean array


print(tt == td)  # same result using the standard operator


print(np.array_equal(tt, td))  # True only if ALL elements match (whole-array check)


print(np.greater(tt, td), np.less(tt, td))  # element-wise > and <


# ======================================================================
# 7. Multi-Dimensional Arrays, Broadcasting, and Reshaping
# ======================================================================

tensor_3d = np.array([[[1, 2, 3, 4], [2, 4, 3, 5], [5, 6, 7, 8], [7, 8, 5, 4]],
                       [[1, 2, 3, 4], [2, 4, 3, 5], [5, 6, 7, 8], [7, 8, 5, 4]]])
print(tensor_3d.shape, tensor_3d.ndim)


# ======================================================================
# Summing along an axis collapses that dimension:
# ======================================================================

np.sum(tensor_3d, axis=0)   # sum across the outermost axis (the two "layers")


np.sum(tensor_3d, axis=1)   # sum across rows within each layer


# ======================================================================
# Broadcasting: NumPy automatically stretches a smaller array to match a larger one's shape when their trailing dimensions are compatible — no explicit loop needed.
# ======================================================================

matrix_4x4 = np.array([[1, 2, 3, 4],
                        [2, 4, 3, 5],
                        [5, 6, 7, 8],
                        [7, 8, 5, 4]])

print(matrix_4x4 + [1, 2, 3, 4])  # the 1-D array is broadcast across every row


# ======================================================================
# Reshaping: `reshape` rearranges the same elements into a new shape (the total element count must stay the same: 4x4 = 16 = 2x8).
# ======================================================================

print(matrix_4x4.reshape(2, 8))


# ======================================================================
# Summary
#
# | Category | Functions covered |
# |---|---|
# | Creation | `np.array`, `np.zeros`, `np.full`, `np.arange`, `np.linspace`, `np.random.random` |
# | Attributes | `.ndim`, `.shape`, `.dtype`, `.size` |
# | Aggregation | `.sum`, `.max`, `.min`, `.mean`, `.std`, `np.argmax`, `np.argmin` |
# | Exponential/Log | `np.exp`, `np.log`, `np.log2`, `np.log10`, `np.sqrt`, `np.power` |
# | Arithmetic | `np.add`, `np.subtract`, `np.multiply`, `np.divide` |
# | Trigonometric | `np.sin`, `np.cos`, `np.tan` |
# | Rounding | `np.round`, `np.floor`, `np.ceil` |
# | Other | `np.abs`, `np.mod` |
# | Comparison | `np.equal`, `np.array_equal`, `np.greater`, `np.less` |
# | Shape ops | `axis`-wise `np.sum`, broadcasting, `.reshape` |
#
# ======================================================================