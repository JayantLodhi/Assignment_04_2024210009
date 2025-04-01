from Linear_algebra_module import Vector
from Linear_algebra_module import Vector_function as vf

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

print(vf.add(v1, v2))  #[5 7 9]
print(vf.dot(v1, v2))  #32
print(vf.cross(v1, v2))  #[-3 6 -3]
print(vf.size(v1))  #3

v3 = Vector([30, 45, 60])

print(vf.cosine(v3))   #[0.8660254  0.70710678 0.5]
print(vf.sine(v3))    #[0.5 0.70710678 0.8660254 ]