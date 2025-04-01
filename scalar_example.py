from Linear_algebra_module import Scalar
from Linear_algebra_module import Scalar_function as sf

s1 = Scalar(2)
s2 = Scalar(3)

print(sf.add(s1, s2)) # 5
print(sf.mult(s1, s2))  # 6
print(sf.pow(s1, s2))  # 8
print(sf.exp(s1))  # 7.38905609893065
print(sf.size(s1)) # 1

s3 = Scalar(4)
s4 = Scalar(5)

#this sum function requires minimum 2 parameter
print(sf.sum(s1, s2, s3, s4)) #14