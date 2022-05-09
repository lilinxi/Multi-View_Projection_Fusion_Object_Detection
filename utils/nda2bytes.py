from io import BytesIO

import numpy as np

def ndarray2bytes(nda: np.ndarray) -> bytes:
    nda_bytes = BytesIO()
    np.save(nda_bytes, nda, allow_pickle=False)
    return nda_bytes.getvalue()


def bytes2ndarray(nda_bytes: bytes) -> np.ndarray:
    return np.load(BytesIO(nda_bytes), allow_pickle=False)

if __name__ == '__main__':
    A = np.array([
        1, 2, 3, 4, 4,
        2, 3, 4, 5, 3,
        4, 5, 6, 7, 2,
        5, 6, 7, 8, 9,
        6, 7, 8, 9, 0]).reshape(5, 5)
    A = A.astype(np.float64)

    bytes_A = ndarray2bytes(A)
    ndarray_A = bytes2ndarray(bytes_A)
    assert np.array_equal(A, ndarray_A)

