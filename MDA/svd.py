import numpy as np
from numpy.linalg import matrix_rank, svd

SIZE = 12

H = np.zeros((SIZE, SIZE))

for i in range(SIZE):
    for j in range(SIZE):
        H[i][j] = 1 / (i + 1 + j + 1 - 1)

u, s, vh = svd(H, full_matrices=False)
print(u.shape, s.shape, vh.shape)
print(matrix_rank(H), matrix_rank(u), matrix_rank(s), matrix_rank(vh))