import numpy as np

class Scalar:                   
    def __init__(self, value):  # Constructor
        self.value = value

    def __str__(self):           # String representation of the object
        return str(self.value)
    
    def add(self, other):                # method to add two Scalar objects
        output = self.value + other.value
        return Scalar(output)
    
    def sum(self, *args):    # args can take n number of arguments and returns tuple of arguments
        output = self.value

        for scalar_obj in args:
            output += scalar_obj.value
        return Scalar(output)
        
    def multiply(self, other):           # method to multiply two Scalar objects
        output = self.value * other.value
        return Scalar(output)
    
    def pow(self, other):              # method to find power of first Scalar object to the second Scalar object 
        output = self.value ** other.value
        return Scalar(output)
    
    def exp(self):              # method to find exponential of the Scalar object
        output = np.exp(self.value)
        return Scalar(output)
    
    @staticmethod      # this static method define a method that is not dependent on the object itself
    def size():   # method to return the size of the Scalar object
        return 1
    
Scalar1 = Scalar(2)
Scalar2 = Scalar(3)

print(Scalar1.add(Scalar2))
print(Scalar1.multiply(Scalar2))
print(Scalar1.pow(Scalar2))
print(Scalar1.exp())
print(Scalar1.size())


Scalar3 = Scalar(4)
Scalar4 = Scalar(5)

print(Scalar1.sum(Scalar2, Scalar3, Scalar4))  

print(Scalar1.sum())



class Vector:
    def __init__(self, values):        # Constructor
        self.values = np.array(values)

    def __str__(self):            # String representation of the object
        return str(self.values)
    
    def add(self, other):                    
        output = np.add(self.values, other.values)      # np.add to add two arrays element-wise
        return Vector(output)
    
    def cosine(self):
        radian_values = np.radians(self.values)    #np.radians to convert the values from degree to radian               
        output = np.cos(radian_values)     #np.cos to find cosine of the array element-wise
        return Vector(output)
    
    def sine(self):
        radian_values = np.radians(self.values)   #np.radians to convert the values from degree to radian
        output = np.sin(radian_values)   #np.sin to find sine of the array element-wise
        return Vector(output)

    def dot(self, other):
        output = np.dot(self.values, other.values)   #np.dot to find dot product of two arrays
        return Scalar(output)
    
    def cross(self, other):
        output = np.cross(self.values, other.values)  #np.cross to find cross product of two arrays
        return Vector(output)
    
    def size(self):
        return np.size(self.values)    #np.size to find the size of the array
    
Vector1 = Vector([1, 2, 3])
Vector2 = Vector([4, 5, 6])

print()
print(Vector1.add(Vector2))
print(Vector1.dot(Vector2))
print(Vector1.cross(Vector2))
print(Vector1.size())

Vector3 = Vector([0, 180])    #must be in degrees
Vector4 = Vector([90, 270])

print(Vector3.cosine())
print(Vector4.sine())


class Matrix:
    def __init__(self, values):       # Constructor
        self.values = np.array(values)

    def __str__(self):        # String representation of the object
        return str(self.values)
    
    def add(self, other):                       
        output = np.add(self.values, other.values)     # np.add to add two matrix element-wise
        return Matrix(output)
    
    def sub(self, other):
        output = np.subtract(self.values, other.values)    # np.subtract to subtract two matrix element-wise
        return Matrix(output)
    
    def mult(self, other):
        output = np.matmul(self.values, other.values)   # np.matmul to do matrix multiplication of two matrix
        return Matrix(output)
    
    def size(self):
        return np.size(self.values)     # np.size to find the size of the matrix
    
Matrix1 = Matrix([[1, 2], [3, 4]])     
Matrix2 = Matrix([[5, 6], [7, 8]])

print()
print(f"{Matrix1.add(Matrix2)}\n")
print(f"{Matrix1.sub(Matrix2)}\n")
print(f"{Matrix1.mult(Matrix2)}\n")
print(f"{Matrix1.size()}\n")