import numpy as np

def load_matrix(file):
    try:
        matrix = np.loadtxt(open(file, "rb"), delimiter=", \t", skiprows=1)
        print(matrix)
        return matrix
    except:
        raise ValueError("Matrix " + file + " could not be loaded! Exiting.")

def read_matrices(folder):
    # folder = "files/"
    file_format = ".csv"
    
    H = load_matrix(folder + "H" + file_format)
    A = load_matrix(folder + "A" + file_format)
    E = load_matrix(folder + "E" + file_format)
    
    print("Loaded matrices")
    return H, A, E
