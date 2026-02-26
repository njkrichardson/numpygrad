from numpy import require
import numpygrad as npg

npg.manual_seed(0)
Log = npg.Log(__name__)


def main():
    a = npg.ones((2, 3), requires_grad=True)
    b = npg.ones((3,), requires_grad=True)
    c = a @ b
    print(c)
    c.backward()
    print(a.grad)
    print(b.grad)


if __name__ == "__main__":
    main()
