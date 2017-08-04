import numpy as np

def main():
    truth = np.ones(5,dtype=int)

    predict = [1, 2, 1, 0, 0]

    print("test")
    print(truth == predict)
    print("Predictions have an accuracy of {:.2f}%.".format((truth == predict).mean()*100))


if __name__ == "__main__":
    main()