from Linear_algebra_module import Vector
from Linear_algebra_module import Vector_function as vf

v1 = Vector([1,2,3])
v2 = Vector([4,5,6])

print(vf.add(v1, v2))
print(vf.cosine(v1))
print(vf.sine(v1))
print(vf.dot(v1, v2))
print(vf.cross(v1, v2))
print(vf.size(v1))