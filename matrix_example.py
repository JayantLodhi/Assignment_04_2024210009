from Linear_algebra_module import Matrix
from Linear_algebra_module import Matrix_function as mf

m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])

print()
print(f"{mf.add(m1, m2)}\n")  # [[6 8][10 12]]
print(f"{mf.sub(m1, m2)}\n")   # [[-4 -4][-4 -4]]
print(f"{mf.mult(m1, m2)}\n")  # [[19 22][43 50]]
print(f"{mf.size(m1)}\n")  # (2,2)

print(f"{mf.trace(m2)}\n")  # 13

Matrix3 = Matrix([[1,2,3],[4,5,6]])

print(f"{mf.transpose(Matrix3)}\n") #[[1 4][2 5][3 6]]