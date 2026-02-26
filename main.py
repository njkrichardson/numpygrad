from numpy import require
import numpygrad as npg

npg.manual_seed(0)
Log = npg.Log(__name__)


def main():
    a = npg.array([2.0, 3.0], requires_grad=True)
    b = npg.array([4.0, 5.0], requires_grad=True)
    # c = a * b
    c = npg.mul(a, b)
    print(f"{c=}")
    c.backward()
    print(f"{a.grad=}")
    print(f"{b.grad=}")


if __name__ == "__main__":
    main()
