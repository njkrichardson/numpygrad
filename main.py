from numpy import require
import numpygrad as npg
from numpygrad.nn import Linear

npg.manual_seed(0)
Log = npg.Log(__name__)


def main():
    batch_size: int = 2
    net = Linear(3, 2)
    x = npg.randn((batch_size, 3))
    out = net(x).sum()
    out.backward()
    print(net.weight.grad)


if __name__ == "__main__":
    main()
