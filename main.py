from numpy import require
import numpygrad as npg

npg.manual_seed(0)
Log = npg.Log(__name__)


def main():
    a = npg.array([2.0, 3.0], requires_grad=True)
    print(a)
    b = a.sum()
    print(b)
    b.backward()
    print(a.grad)


if __name__ == "__main__":
    main()
