### Exercise 1: 

import numpy as np

# 1D 
array_1d = np.array([5, 10, 15, 20, 25], dtype=np.float64)
print("1D Array:", array_1d)

# 2D 
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array Shape:", array_2d.shape)
print("2D Array Size:", array_2d.size)

# 3D 
array_3d = np.random.rand(2, 3, 4)
print("3D Array Dimensions:", array_3d.ndim)
print("3D Array Shape:", array_3d.shape)


### Exercise 2: 


# 1. 
array_1d = np.arange(10)
array_reversed = array_1d[::-1]
print("Reversed 1D Array:", array_reversed)

# 2. 
array_2d = np.arange(12).reshape(3, 4)
subarray = array_2d[:2, -2:]  # First two rows, last two columns
print("Extracted Subarray:\n", subarray)

# 3. 
array_random = np.random.randint(0, 11, (5, 5))
array_modified = np.where(array_random > 5, 0, array_random)  # Replace elements >5 with 0
print("Original Array:\n", array_random)
print("Modified Array:\n", array_modified)


### Exercise 3: 

# 1. 
identity_matrix = np.eye(3)
print("Identity Matrix:\n", identity_matrix)
print("Number of Dimensions (ndim):", identity_matrix.ndim)
print("Shape:", identity_matrix.shape)
print("Size (Total elements):", identity_matrix.size)
print("Item Size (Bytes per element):", identity_matrix.itemsize)
print("Total Bytes (nbytes):", identity_matrix.nbytes)

# 2. 
linspace_array = np.linspace(0, 5, 10)
print("\nLinspace Array:", linspace_array)
print("Data Type:", linspace_array.dtype)

# 3. 
array_3d = np.random.randn(2, 3, 4)
print("\n3D Array:\n", array_3d)
print("Sum of All Elements:", np.sum(array_3d))

### Exercise 4: 

# 1. 
array_1d = np.random.randint(0, 50, 20)
selected_elements = array_1d[[2, 5, 7, 10, 15]]  # Fancy indexing
print("Original 1D Array:", array_1d)
print("Selected Elements:", selected_elements)

# 2.
array_2d = np.random.randint(0, 30, (4, 5))
masked_elements = array_2d[array_2d > 15]  # Boolean mask for values > 15
print("\nOriginal 2D Array:\n", array_2d)
print("Elements Greater Than 15:", masked_elements)

# 3. 
array_negative = np.random.randint(-10, 10, 10)
array_negative[array_negative < 0] = 0  # Setting negative values to zero
print("\nOriginal 1D Array with Negatives:", array_negative)
print("Modified Array (Negatives Set to Zero):", array_negative)

### Exercise 5: 
# 1. 
array1 = np.random.randint(0, 10, 5)
array2 = np.random.randint(0, 10, 5)
concatenated_array = np.concatenate((array1, array2))
print("Array 1:", array1)
print("Array 2:", array2)
print("Concatenated Array:", concatenated_array)

# 2. 
array_2d = np.random.randint(0, 10, (6, 4))
split_arrays_row = np.split(array_2d, 2, axis=0)  # Splitting along rows
print("\nOriginal 2D Array (6x4):\n", array_2d)
print("First Split Part:\n", split_arrays_row[0])
print("Second Split Part:\n", split_arrays_row[1])

# 3. 
array_2d_col = np.random.randint(0, 10, (3, 6))
split_arrays_col = np.split(array_2d_col, 3, axis=1)  # Splitting along columns
print("\nOriginal 2D Array (3x6):\n", array_2d_col)
print("First Column Split:\n", split_arrays_col[0])
print("Second Column Split:\n", split_arrays_col[1])
print("Third Column Split:\n", split_arrays_col[2])

### Exercise 6: 

# 1. 
array_1d = np.random.randint(1, 101, 15)
mean_value = np.mean(array_1d)
median_value = np.median(array_1d)
std_dev = np.std(array_1d)
variance = np.var(array_1d)

print("1D Array:", array_1d)
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# 2. 
array_2d = np.random.randint(1, 51, (4, 4))
row_sums = np.sum(array_2d, axis=1)
col_sums = np.sum(array_2d, axis=0)

print("\n2D Array:\n", array_2d)
print("Sum of Each Row:", row_sums)
print("Sum of Each Column:", col_sums)

# 3. 
array_3d = np.random.randint(1, 21, (2, 3, 4))
max_along_axis0 = np.max(array_3d, axis=0)
max_along_axis1 = np.max(array_3d, axis=1)
max_along_axis2 = np.max(array_3d, axis=2)

min_along_axis0 = np.min(array_3d, axis=0)
min_along_axis1 = np.min(array_3d, axis=1)
min_along_axis2 = np.min(array_3d, axis=2)

print("\n3D Array:\n", array_3d)
print("Max along Axis 0:\n", max_along_axis0)
print("Max along Axis 1:\n", max_along_axis1)
print("Max along Axis 2:\n", max_along_axis2)

print("\nMin along Axis 0:\n", min_along_axis0)
print("Min along Axis 1:\n", min_along_axis1)
print("Min along Axis 2:\n", min_along_axis2)

### Exercise 7: 

# 1. 
array_1d = np.arange(1, 13)  # Numbers from 1 to 12
array_2d = array_1d.reshape(3, 4)  # Reshape to (3, 4)
print("Reshaped 2D Array (3x4):\n", array_2d)

# 2. 
array_2d_random = np.random.randint(1, 11, (3, 4))
array_transposed = array_2d_random.T  # Transpose the array
print("\nOriginal 2D Array (3x4):\n", array_2d_random)
print("Transposed Array (4x3):\n", array_transposed)

# 3. Create a 2D NumPy array of shape (2, 3) with random integers between 1 and 10
array_2d_small = np.random.randint(1, 11, (2, 3))
array_flattened = array_2d_small.flatten()  # Flatten the array to 1D
print("\nOriginal 2D Array (2x3):\n", array_2d_small)
print("Flattened 1D Array:\n", array_flattened)

### Exercise 8: Broadcasting and Vectorized Operations

# 1.
array_2d = np.random.randint(1, 11, (3, 4))
column_means = np.mean(array_2d, axis=0)  # Mean of each column
normalized_array = array_2d - column_means  # Broadcasting subtraction
print("Original 2D Array:\n", array_2d)
print("Column Means:\n", column_means)
print("Array After Subtracting Column Means:\n", normalized_array)

# 2. Create two 1D NumPy arrays of length 4 and compute the outer product using br
array1 = np.random.randint(1, 5, 4)
array2 = np.random.randint(1, 5, 4)
outer_product = np.outer(array1, array2)  # Outer product using broadcasting
print("\nArray 1:", array1)
print("Array 2:", array2)
print("Outer Product:\n", outer_product)

# 3. Create a 2D NumPy array of shape (4, 5) and add 10 to all elements greater than 5
array_2d_large = np.random.randint(1, 11, (4, 5))
modified_array = np.where(array_2d_large > 5, array_2d_large + 10, array_2d_large)  # Vectorized operation
print("\nOriginal 2D Array:\n", array_2d_large)
print("Modified Array (Adding 10 to elements > 5):\n", modified_array)

### Exercise 9: Sorting and Searching Arrays

# 1. Create a 1D NumPy array with random integers between 1 and 20 of size 10 and sort it
array_1d = np.random.randint(1, 21, 10)
sorted_array = np.sort(array_1d)
print("Original 1D Array:", array_1d)
print("Sorted Array:", sorted_array)

# 2. Create a 2D NumPy array of shape (3, 5) with random integers between 1 and 50
array_2d = np.random.randint(1, 51, (3, 5))
sorted_by_second_col = array_2d[array_2d[:, 1].argsort()]  # Sort by second column
print("\nOriginal 2D Array:\n", array_2d)
print("Array Sorted by Second Column:\n", sorted_by_second_col)

# 3. Create a 1D NumPy array with random integers between 1 and 100 of size 15
array_large = np.random.randint(1, 101, 15)
indices_greater_than_50 = np.where(array_large > 50)[0]  # Get indices of elements > 50
print("\n1D Array:", array_large)
print("Indices of Elements Greater Than 50:", indices_greater_than_50)

### Exercise 10: Linear Algebra with NumPy

# 1. Create a 2D NumPy array of shape (2, 2) and compute the determinant
array_2x2 = np.random.randint(1, 11, (2, 2))
determinant = np.linalg.det(array_2x2)
print("2x2 Matrix:\n", array_2x2)
print("Determinant:", determinant)

# 2. Create a 2D NumPy array of shape (3, 3) and compute eigenvalues and eigenvectors
array_3x3 = np.random.randint(1, 6, (3, 3))
eigenvalues, eigenvectors = np.linalg.eig(array_3x3)
print("\n3x3 Matrix:\n", array_3x3)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# 3. Create two 2D NumPy arrays of shape (2, 3) and (3, 2) and compute matrix product
array_2x3 = np.random.randint(1, 11, (2, 3))
array_3x2 = np.random.randint(1, 11, (3, 2))
matrix_product = np.dot(array_2x3, array_3x2)
print("\nMatrix 2x3:\n", array_2x3)
print("Matrix 3x2:\n", array_3x2)
print("Matrix Product (2x2):\n", matrix_product)

### Exercise 11: Random Sampling and Distributions

import matplotlib.pyplot as plt

# 1. Create a 1D NumPy array of 10 random samples from a uniform distribution over [0, 1)
uniform_samples = np.random.uniform(0, 1, 10)
print("Uniform Distribution Samples:\n", uniform_samples)

# 2. Create a 2D NumPy array of shape (3, 3) with random samples from a normal distribution (mean=0, std=1)
normal_samples = np.random.normal(0, 1, (3, 3))
print("\nNormal Distribution Samples:\n", normal_samples)

# 3. Create a 1D NumPy array of 20 random integers between 1 and 100
random_integers = np.random.randint(1, 101, 20)

# Compute and print the histogram with 5 bins
hist_values, bin_edges = np.histogram(random_integers, bins=5)
print("\nRandom Integers:\n", random_integers)
print("Histogram Values:\n", hist_values)
print("Bin Edges:\n", bin_edges)

# Plot the histogram
plt.hist(random_integers, bins=5, edgecolor='black', alpha=0.7)
plt.xlabel("Value Range")
plt.ylabel("Frequency")
plt.title("Histogram of Random Integers")
plt.show()

### Exercise 12: Advanced Indexing and Selection

# 1. Create a 2D NumPy array of shape (5, 5) with random integers between 1 and 20
array_5x5 = np.random.randint(1, 21, (5, 5))
diagonal_elements = np.diag(array_5x5)
print("Original 5x5 Array:\n", array_5x5)
print("Diagonal Elements:\n", diagonal_elements)

# 2. Create a 1D NumPy array of 10 random integers between 1 and 50 and select prime numbers
array_1d = np.random.randint(1, 51, 10)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

prime_numbers = array_1d[np.vectorize(is_prime)(array_1d)]
print("\nOriginal 1D Array:\n", array_1d)
print("Prime Numbers:\n", prime_numbers)

# 3. Create a 2D NumPy array of shape (4, 4) with random integers between 1 and 10
array_4x4 = np.random.randint(1, 11, (4, 4))
even_numbers = array_4x4[array_4x4 % 2 == 0]  # Selecting even numbers
print("\nOriginal 4x4 Array:\n", array_4x4)
print("Even Numbers:\n", even_numbers)

### Exercise 13: Handling Missing Data

# 1. Create a 1D NumPy array of length 10 with random integers between 1 and 10
array_1d = np.random.randint(1, 11, 10).astype(float)  # Convert to float to allow np.nan
nan_indices = np.random.choice(10, size=3, replace=False)  # Random positions for NaN
array_1d[nan_indices] = np.nan
print("1D Array with NaN values:\n", array_1d)

# 2. Create a 2D NumPy array of shape (3, 4) with random integers between 1 and 10
array_2d = np.random.randint(1, 11, (3, 4)).astype(float)  # Convert to float for NaN support
array_2d[array_2d < 5] = np.nan  # Replace values < 5 with NaN
print("\n2D Array with NaN values:\n", array_2d)

# 3. Create a 1D NumPy array of length 15 with random integers between 1 and 20
array_1d_15 = np.random.randint(1, 21, 15).astype(float)  # Convert to float
nan_indices_15 = np.random.choice(15, size=4, replace=False)  # Random NaN positions
array_1d_15[nan_indices_15] = np.nan

# Identify indices of NaN values
nan_positions = np.where(np.isnan(array_1d_15))[0]
print("\n1D Array with NaN values:\n", array_1d_15)
print("Indices of NaN values:", nan_positions)

### Exercise 14: Performance Optimization with NumPy

import time

# 1. Create a large 1D NumPy array with 1 million random integers between 1 and 100
array_large_1d = np.random.randint(1, 101, 1_000_000)

# Measure time for mean and standard deviation computation
start_time = time.time()
mean_value = np.mean(array_large_1d)
std_dev_value = np.std(array_large_1d)
end_time = time.time()
print("Mean:", mean_value, "Standard Deviation:", std_dev_value)
print("Time taken for mean and std dev:", end_time - start_time, "seconds")

# 2. Create two large 2D NumPy arrays of shape (1000, 1000) with random integers between 1 and 10
array_2d_a = np.random.randint(1, 11, (1000, 1000))
array_2d_b = np.random.randint(1, 11, (1000, 1000))

# Measure time for element-wise addition
start_time = time.time()
array_sum = array_2d_a + array_2d_b  # Element-wise addition
end_time = time.time()
print("\nTime taken for element-wise addition:", end_time - start_time, "seconds")

# 3. Create a 3D NumPy array of shape (100, 100, 100) with random integers between 1 and 10
array_3d = np.random.randint(1, 11, (100, 100, 100))

# Measure time for sum along each axis
start_time = time.time()
sum_axis_0 = np.sum(array_3d, axis=0)
sum_axis_1 = np.sum(array_3d, axis=1)
sum_axis_2 = np.sum(array_3d, axis=2)
end_time = time.time()
print("\nTime taken for summing along axes:", end_time - start_time, "seconds")

### Exercise 15: Cumulative and Aggregate Functions

# 1. Create a 1D NumPy array with the numbers from 1 to 10 and compute cumulative sum and product
array_1d = np.arange(1, 11)
cumsum_array = np.cumsum(array_1d)  # Cumulative sum
cumprod_array = np.cumprod(array_1d)  # Cumulative product
print("1D Array:", array_1d)
print("Cumulative Sum:", cumsum_array)
print("Cumulative Product:", cumprod_array)

# 2. Create a 2D NumPy array of shape (4, 4) with random integers between 1 and 20
array_2d = np.random.randint(1, 21, (4, 4))
cumsum_rows = np.cumsum(array_2d, axis=1)  # Cumulative sum along rows
cumsum_cols = np.cumsum(array_2d, axis=0)  # Cumulative sum along columns
print("\n2D Array:\n", array_2d)
print("Cumulative Sum Along Rows:\n", cumsum_rows)
print("Cumulative Sum Along Columns:\n", cumsum_cols)

# 3. Create a 1D NumPy array with 10 random integers between 1 and 50
array_random = np.random.randint(1, 51, 10)
min_value = np.min(array_random)
max_value = np.max(array_random)
sum_value = np.sum(array_random)
print("\nRandom 1D Array:", array_random)
print("Minimum Value:", min_value)
print("Maximum Value:", max_value)
print("Sum of Elements:", sum_value)

### Exercise 16: Working with Dates and Times

# 1. Create an array of 10 dates starting from today with a daily frequency
dates_daily = np.arange(np.datetime64('today'), np.datetime64('today') + 10, dtype='datetime64[D]')
print("Array of 10 Dates (Daily Frequency):\n", dates_daily)

# 2. Create an array of 5 dates starting from January 1, 2022 with a monthly frequency
dates_monthly = np.arange(np.datetime64('2022-01-01'), np.datetime64('2022-06-01'), dtype='datetime64[M]')
print("\nArray of 5 Dates (Monthly Frequency):\n", dates_monthly)

# 3. Create a 1D array with 10 random timestamps in the year 2023 and convert to datetime64
random_days = np.random.randint(0, 365, 10)  # Generate 10 random days in the year
timestamps = np.datetime64('2023-01-01') + random_days  # Add random days to start of year
print("\nArray of 10 Random Timestamps in 2023:\n", timestamps)

### Exercise 17: Creating Arrays with Custom Data Types

# 1. Create a 1D NumPy array with custom dtype storing integers and their binary representation
custom_dtype = np.dtype([('number', np.int32), ('binary', 'U10')])
array_custom = np.array([(5, bin(5)[2:]), (10, bin(10)[2:]), (15, bin(15)[2:]), (20, bin(20)[2:]), (25, bin(25)[2:])], dtype=custom_dtype)
print("Custom Array (Integers and Binary Representation):\n", array_custom)

# 2. Create a 2D NumPy array with custom dtype storing complex numbers
complex_dtype = np.complex128
array_complex = np.array([[1+2j, 3+4j, 5+6j],
                          [7+8j, 9+10j, 11+12j],
                          [13+14j, 15+16j, 17+18j]], dtype=complex_dtype)
print("\n2D Array with Complex Numbers:\n", array_complex)

# 3. Create a structured array to store book information
book_dtype = np.dtype([('title', 'U50'), ('author', 'U50'), ('pages', np.int32)])
books = np.array([
    ("The Great Gatsby", "F. Scott Fitzgerald", 180),
    ("1984", "George Orwell", 328),
    ("To Kill a Mockingbird", "Harper Lee", 281)
], dtype=book_dtype)
print("\nStructured Array for Books:\n", books)
