import numpy as np

from libft.preprocessing import StandardScaler

if __name__ == '__main__':
    sc = StandardScaler()
    data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    print(sc.fit(data))
    print(sc.mean)
    print(sc.transform(data))
    print(sc.transform([[2, 2]]))
    print(sc.inverse_transform(sc.transform(data)))
