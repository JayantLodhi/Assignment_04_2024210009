import numpy as np

class Scalar:            
    def __init__(self, value):    # Constructor
        val = np.array(value)

        if val.ndim == 0:       #val.ndim == 0 because dimension of np.array(int/float) will be 0.
            self.value = val
        else:
            raise ValueError("Argument must be integar or float")  

    def __str__(self):      # String representation of the object
        return str(self.value)
    
class Vector:
    def __init__(self, value):    # Constructor
        val = np.array(value)

        if val.ndim == 1:      #val.ndim == 1 because dimension of np.array(1d list) will be 1.
            self.value = val
        else: 
            raise ValueError("Argument must be vector (1 dimension list)")  

    def __str__(self):      # String representation of the object
        return str(self.value)

class Matrix:
    def __init__(self, value):    # Constructor
        val = np.array(value)

        if val.ndim == 2:              #val.ndim == 2 because dimension of np.array(2d list)) will be 2.
            self.value = val
        else:
            raise ValueError("Argument must be matrix (2 dimensional list)")   

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

            if np.array(output).ndim == 0:
                return Scalar(output)
            else:
                return Vector(output)
        
        else:
            raise ValueError("Both argument must be of type Vector")
    
    @staticmethod
    def size(vector):    # to find the size of the array

        if isinstance(vector, Vector):
            return np.size(vector.value)
        
        else:
            raise ValueError("Argument must be of type Vector")


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
            return np.shape(matrix.value)
        
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