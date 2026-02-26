from numpy import require
import numpygrad as npg
from numpygrad.nn import Linear

npg.manual_seed(0)
Log = npg.Log(__name__)


def main():
    batch_size: int = 2
    # net = Linear(3, 2)
    x = npg.randn((batch_size, 3), requires_grad=True)
    y = x.reshape(-1)
    out = y.sum()
    out.backward()
    print(x.shape)
    print(y.shape)
    print(x.grad)
    # out = net(x).sum()
    # out.backward()
    # print(net.weight.grad)


if __name__ == "__main__":
    main()
