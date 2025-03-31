import numpy as np

class Scalar:            
    def __init__(self, value):    # Constructor
        val = np.array(value)

        if val.ndim == 0:       #val.ndim == 0 because dimension of np.array(int/float) will be 0.
            self.value = val
        else:
            raise ValueError("Argument must be integar or float")  # this will raise error

    def __str__(self):      # String representation of the object
        return str(self.value)
    
class Vector:
    def __init__(self, value):    # Constructor
        val = np.array(value)

        if val.ndim == 1:      #val.ndim == 1 because dimension of np.array(1d list) will be 1.
            self.value = val
        else: 
            raise ValueError("Argument must be vector (1 dimension list)")  # this will raise error

    def __str__(self):      # String representation of the object
        return str(self.value)

class Matrix:
    def __init__(self, value):    # Constructor
        val = np.array(value)

        if val.ndim == 2:              #val.ndim == 2 because dimension of np.array(2d list)) will be 2.
            self.value = val
        else:
            raise ValueError("Argument must be matrix (2 dimensional list)")    # this will raise error

    def __str__(self):      # String representation of the object
        return str(self.value)
    

class Scalar_function:  

    @staticmethod               
    def add(scalar1, scalar2):                   # method to add two Scalar objects 

        if isinstance(scalar1, Scalar) and isinstance(scalar2, Scalar):
            
            output = scalar1.value + scalar2.value
            return Scalar(output)
        
        else:
            raise ValueError("Both Argument must be of type Scalar")
    
    @staticmethod                        # method to add multiple Scalar objects (require minimum 2 arguments)
    def sum(scalar1, scalar2, *args):   # args can take n number of arguments and returns tuple of arguments
        
        if isinstance(scalar1, Scalar) and isinstance(scalar2, Scalar):
            output = scalar1.value + scalar2.value  
        else:
            raise ValueError("All argument must be of type Scalar")
        
        for scalar_obj in args:

            if isinstance(scalar_obj, Scalar): 
                output += scalar_obj.value
            else:
                raise ValueError("All argument must be of type Scalar")
            
        return Scalar(output)

    @staticmethod
    def mult(scalar1, scalar2):       # method to multiply two Scalar objects

        if isinstance(scalar1, Scalar) and isinstance(scalar2, Scalar):
            
            output = scalar1.value * scalar2.value
            return Scalar(output)
        
        else:
            raise ValueError("Both Argument must be of type Scalar")
    
    @staticmethod
    def pow(scalar1, scalar2):           # method to find power of first Scalar object to the second Scalar object

        if isinstance(scalar1, Scalar) and isinstance(scalar2, Scalar):

            output = scalar1.value ** scalar2.value
            return Scalar(output)
        
        else:
            raise ValueError("Both Argument must be of type Scalar")
    
    @staticmethod
    def exp(scalar):         # method to find exponential of the Scalar object

        if isinstance(scalar, Scalar):

            output = np.exp(scalar.value)
            return Scalar(output)
        
        else:
            raise ValueError("Argument must be of type Scalar")
    
    @staticmethod
    def size(scalar):       # method to return the size of the Scalar object

        if isinstance(scalar, Scalar):
            return np.size(scalar.value)
        
        else:
            raise ValueError("Argument must be of type Scalar")
    
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

        if isinstance(vector1, Vector) and isinstance(vector2, Vector):

            output = np.add(vector1.value, vector2.value)
            return Vector(output)
        
        else:
            raise ValueError("Both argument must be of type Vector")
    
    @staticmethod
    def cosine(vector):  # to find cosine of the array element-wise

        if isinstance(vector, Vector):
            
            radian_value = np.radians(vector.value)  # Convert degrees to radians
            output = np.cos(radian_value)
            return Vector(output)
        
        else:
            raise ValueError("Argument must be of type Vector")

    @staticmethod
    def sine(vector):    # to find sine of the array element-wise

        if isinstance(vector, Vector):

            radian_value = np.radians(vector.value)  # Convert degrees to radians
            output = np.sin(radian_value)
            return Vector(output)
        
        else:
            raise ValueError("Argument must be of type Vector")
    
    @staticmethod
    def dot(vector1, vector2):   # to find dot product of two array

        if isinstance(vector1, Vector) and isinstance(vector2, Vector):
            
            output = np.dot(vector1.value, vector2.value)
            return Scalar(output)
    
        else:
            raise ValueError("Both argument must be of type Vector")
        
    @staticmethod
    def cross(vector1, vector2):  # to find cross product of two array

        if isinstance(vector1, Vector) and isinstance(vector2, Vector):
            
            output = np.cross(vector1.value, vector2.value)
            return Vector(output)
        
        else:
            raise ValueError("Both argument must be of type Vector")
    
    @staticmethod
    def size(vector):    # to find the size of the array

        if isinstance(vector, Vector):
            return np.size(vector.value)
        
        else:
            raise ValueError("Argument must be of type Vector")
    

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

        if isinstance(matrix1, Matrix) and isinstance(matrix2, Matrix):     
        
            output = np.add(matrix1.value, matrix2.value)
            return Matrix(output)
        
        else:
            raise ValueError("Both argument must be of type Matrix")
    
    @staticmethod
    def sub(matrix1, matrix2):  # to sub two matrix objects

        if isinstance(matrix1, Matrix) and isinstance(matrix2, Matrix):
        
            output = np.subtract(matrix1.value, matrix2.value)
            return Matrix(output)
        
        else:
            raise ValueError("Both argument must be of type Matrix")
    
    @staticmethod
    def mult(matrix1, matrix2):       # method to multiply two Matrix objects

        if isinstance(matrix1, Matrix) and isinstance(matrix2, Matrix):

            output = np.matmul(matrix1.value, matrix2.value)
            return Matrix(output)
        
        else:
            raise ValueError("Both argument must be of type Matrix")
    
    @staticmethod
    def size(matrix):       # method to return the size of the Matrix object
        
        if isinstance(matrix, Matrix):
            return np.size(matrix.value)
        
        else:
            raise ValueError("Argument must be of type Matrix")
        
    @staticmethod
    def trace(matrix):         # Method to find transpose of the Matrix object

        if isinstance(matrix, Matrix):
            shape = np.shape(matrix.value)
            row, column = shape   

            if row == column:        
                trace = 0  #initially zero

                for i in range(0, shape[0]):
                    trace += matrix.value[i,i]   #increasing trace by matrix[row=column] in each iteration
                    
                return Scalar(trace)
            
            else:
                raise ValueError("Matrix must be square")

        else:
            raise ValueError("Argument must be of type Matrix")
        
    @staticmethod
    def transpose(matrix):

        if isinstance(matrix, Matrix):

            shape = np.shape(matrix.value)
            row, column = shape

            tran_mat = np.empty((column, row))   #initially empty matrix with reverse shape

            for r in range(0, row):
                for c in range(0, column):
                    tran_mat[c,r] = matrix.value[r,c]    #substituting transposed value
                     
            return Matrix(tran_mat)

        else:
            raise ValueError("Argument must be of type Matrix")
    
Matrix1 = Matrix([[1, 2], [3, 4]])
Matrix2 = Matrix([[5, 6], [7, 8]])

print()
print(f"{Matrix_function.add(Matrix1, Matrix2)}\n")  # [[6 8][10 12]]
print(f"{Matrix_function.sub(Matrix1, Matrix2)}\n")   # [[-4 -4][-4 -4]]
print(f"{Matrix_function.mult(Matrix1, Matrix2)}\n")  # [[19 22][43 50]]
print(f"{Matrix_function.size(Matrix1)}\n")  # 4

print(f"{Matrix_function.trace(Matrix2)}\n")  # 13

Matrix3 = Matrix([[1,2,3],[4,5,6]])

print(f"{Matrix_function.transpose(Matrix3)}\n")
