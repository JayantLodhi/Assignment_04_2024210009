import numpy as np

class Entities:            # base class for all entities
    def __init__(self, value):
        self.value = np.array(value)

    def __str__(self):      # String representation of the object
        return str(self.value)
    
class Scalar(Entities):  # scalar inherit from Entities class
    pass

class Vector(Entities):  # vector inherit from Entities class
    pass   

class Matrix(Entities):  # matrix inherit from Entities class
    pass


class Scalar_function:  

    @staticmethod               
    def add(scalar1, scalar2):                   # method to add two Scalar objects     
        output = scalar1.value + scalar2.value
        return Scalar(output)
    
    @staticmethod                        # method to add multiple Scalar objects (require minimum 2 arguments)
    def sum(scalar1, scalar2, *args):   # args can take n number of arguments and returns tuple of arguments
        output = scalar1.value + scalar2.value  

        for scalar_obj in args:
            output += scalar_obj.value
        return Scalar(output)

    @staticmethod
    def mult(scalar1, scalar2):       # method to multiply two Scalar objects
        output = scalar1.value * scalar2.value
        return Scalar(output)
    
    @staticmethod
    def pow(scalar1, scalar2):           # method to find power of first Scalar object to the second Scalar object
        output = scalar1.value ** scalar2.value
        return Scalar(output)
    
    @staticmethod
    def exp(scalar):         # method to find exponential of the Scalar object
        output = np.exp(scalar.value)
        return Scalar(output)
    
    @staticmethod
    def size(scalar1):       # method to return the size of the Scalar object
        return np.size(scalar1.value)
    
Scalar1 = Scalar(2)
Scalar2 = Scalar(3)

print(Scalar_function.add(Scalar1, Scalar2)) # 5
print(Scalar_function.mult(Scalar1, Scalar2))  # 6
print(Scalar_function.pow(Scalar1, Scalar2))  # 8
print(Scalar_function.exp(Scalar1))  # 7.38905609893065
print(Scalar_function.size(Scalar1)) # 1

Scalar3 = Scalar(4)
Scalar4 = Scalar(5)

#this sum function requires minimum 2 parameter
print(Scalar_function.sum(Scalar1, Scalar2, Scalar3, Scalar4)) #14


class Vector_function:  

    @staticmethod               
    def add(vector1, vector2):                   # method to add two Vector objects     
        output = np.add(vector1.value, vector2.value)
        return Vector(output)
    
    @staticmethod
    def cosine(vector):  # to find cosine of the array element-wise
        radian_value = np.radians(vector.value)  # Convert degrees to radians
        output = np.cos(radian_value)
        return Vector(output)
    
    @staticmethod
    def sine(vector):    # to find sine of the array element-wise
        radian_value = np.radians(vector.value)  # Convert degrees to radians
        output = np.sin(radian_value)
        return Vector(output)
    
    @staticmethod
    def dot(vector1, vector2):   # to find dot product of two array
        output = np.dot(vector1.value, vector2.value)
        return Scalar(output)
    
    @staticmethod
    def cross(vector1, vector2):  # to find cross product of two array
        output = np.cross(vector1.value, vector2.value)
        return Vector(output)
    
    @staticmethod
    def size(vector):    # to find the size of the array
        return np.size(vector.value)
    

Vector1 = Vector([1, 2, 3])
Vector2 = Vector([4, 5, 6])

print()
print(Vector_function.add(Vector1, Vector2))   # [5 7 9]
print(Vector_function.dot(Vector1, Vector2))   # 32
print(Vector_function.cross(Vector1, Vector2)) # [-3  6 -3]
print(Vector_function.size(Vector1))  #3

Vector3 = Vector([30, 45, 60])

print(Vector_function.cosine(Vector3))  # [0.8660254  0.70710678 0.5]
print(Vector_function.sine(Vector3))  # [0.5 0.70710678 0.8660254 ]


class Matrix_function:

    @staticmethod
    def add(matrix1, matrix2):                   # method to add two Matrix objects     
        output = np.add(matrix1.value, matrix2.value)
        return Matrix(output)
    
    @staticmethod
    def sub(matrix1, matrix2):  # to sub two matrix objects
        output = np.subtract(matrix1.value, matrix2.value)
        return Matrix(output)
    
    @staticmethod
    def mult(matrix1, matrix2):       # method to multiply two Matrix objects
        output = np.matmul(matrix1.value, matrix2.value)
        return Matrix(output)
    
    @staticmethod
    def size(matrix):       # method to return the size of the Matrix object
        return np.size(matrix.value)
    
Matrix1 = Matrix([[1, 2], [3, 4]])
Matrix2 = Matrix([[5, 6], [7, 8]])

print()
print(f"{Matrix_function.add(Matrix1, Matrix2)}\n")  # [[6 8][10 12]]
print(f"{Matrix_function.sub(Matrix1, Matrix2)}\n")   # [[-4 -4][-4 -4]]
print(f"{Matrix_function.mult(Matrix1, Matrix2)}\n")  # [[19 22][43 50]]
print(f"{Matrix_function.size(Matrix1)}\n")  # 4
