import numpygrad as npg

npg.manual_seed(0)

Log = npg.Log(__name__)


def main():
    x = npg.ones((3, 3), requires_grad=True)
    b = npg.ones((3,), requires_grad=True)
    y = npg.setitem(x, (slice(None), 0), b)

    out = y.sum()
    out.backward()

    print(x.grad)


if __name__ == "__main__":
    main()
