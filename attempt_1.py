import numpy as np

class Scalar:                   
    def __init__(self, value):  # Constructor
        self.value = value

    def __str__(self):           # String representation of the object
        return str(self.value)
    
    def add(self, other):                # method to add two Scalar objects
        output = self.value + other.value
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