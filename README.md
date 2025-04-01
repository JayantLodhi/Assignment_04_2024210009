# 📐 Linear Algebra Module

This Python module provides fundamental operations for working with **Scalars, Vectors, and Matrices**, using the **NumPy** library for efficient calculations.

---

## 📝 Features
- **Scalars**: Addition, multiplication, exponentiation, sum, and exponential functions.
- **Vectors**: Addition, dot product, cross product, trigonometric functions.
- **Matrices**: Addition, subtraction, multiplication, transpose, trace, and size calculations.

---

## 🚀 Installation
Make sure you have **Python** installed. Then, install **NumPy**:

```sh
pip install numpy
```

---

## 🏗️ How It Works
1) The script allows users to perform basic linear algebra operations on `scalars`, `vectors`, and `matrices`.

2) Users can create objects for scalars, vectors, or matrices and then apply various mathematical functions.
   - Scalars are represented as single numeric values.
   - Vectors are lists or NumPy arrays.
   - Matrices are 2D lists or NumPy arrays.
  
3) The module utilizes **NumPy** for efficient computations.
   
4) Functions are structured to handle common mathematical operations, such as addition, multiplication, and transformations.
   
5) Results can be used in further calculations or printed for verification.

---

## 📝 Code Overview

### **Scalar Operations**
#### `add(s1, s2)`
- Adds two scalar objects.

#### `mult(s1, s2)`
- Multiplies two scalars.

#### `pow(s1, s2)`
- Computes power (s1^s2).

#### `exp(s1)`
- Computes the exponential of a scalar.

#### `sum(*scalars)`
- Computes the sum of multiple scalars.

#### `size(scalar)`
- Returns the size of scalar.

**Example:**
```python
from Linear_algebra_module import Scalar, Scalar_function as sf
s1 = Scalar(2)
s2 = Scalar(3)
print(sf.add(s1, s2))  # Output: 5
print(sf.exp(s1))  # Output: 7.389
```

---

### **Vector Operations**
#### `add(v1, v2)`
- Performs element-wise addition.

#### `dot(v1, v2)`
- Computes the dot product between two vectors.

#### `cross(v1, v2)`
- Computes the cross product between two vectors.

#### `cosine(v)`
- Computes element-wise cosine of a vector.

#### `sine(v)`
- Computes element-wise sine of a vector.

#### `size(vector)`
- Return the size of vector (no. of elements).

**Example:**
```python
from Linear_algebra_module import Vector, Vector_function as vf
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])
print(vf.dot(v1, v2))  # Output: 32
print(vf.cross(v1, v2))  # Output: [-3, 6, -3]
```

---

### **Matrix Operations**
#### `add(m1, m2)`
- Performs element-wise addition between two matrix.

#### `sub(m1, m2)`
- Performs element-wise subtraction between two matrix.

#### `mult(m1, m2)`
- Computes matrix multiplication between two matrix.

#### `size(m)`
- Retrieves the size of the matrix (m x n).

#### `trace(m)`
- Computes the sum of diagonal elements of the matrix.

#### `transpose(m)`
- Returns the transpose of a matrix.

**Example:**
```python
from Linear_algebra_module import Matrix, Matrix_function as mf
m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])
print(mf.add(m1, m2))  # Output: [[6, 8], [10, 12]]
print(mf.transpose(m1))  # Output: [[1, 3], [2, 4]]
```

---

## 📌 Notes
- This module is for educational purposes only.
- Ensure NumPy is installed before running the script.

---

## 👨‍💻 Author
**Jayant Lodhi**  (2024210009)
