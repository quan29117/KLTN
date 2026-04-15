import numpy as np
import pandas as pd

def main():
    print("=== TEST NUMPY ===")
    arr = np.array([1, 2, 3, 4, 5])
    print("Array:", arr)
    print("Mean:", np.mean(arr))

    print("\n=== TEST PANDAS ===")
    data = {
        "Name": ["An", "Binh", "Chi"],
        "Age": [20, 21, 22]
    }
    df = pd.DataFrame(data)
    print(df)

    print("\n=== XỬ LÝ DATA ===")
    print("Tuổi trung bình:", df["Age"].mean())

if __name__ == "__main__":
    main()