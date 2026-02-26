from numpy import require
import numpygrad as npg

npg.manual_seed(0)
Log = npg.Log(__name__)


def main():
    a = npg.randn((2, 3), requires_grad=True)
    b = npg.relu(a)
    c = b.sum()

    print(a)
    print(b)

    c.backward()

    print(a.grad)


if __name__ == "__main__":
    main()
