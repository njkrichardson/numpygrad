from numpy import require
import numpygrad as npg
from numpygrad.nn import Linear, MLP

npg.manual_seed(0)
Log = npg.Log(__name__)


def main():
    hidden_sizes = [4, 3]
    input_dim = 3
    output_dim = 1
    net = MLP(input_dim, hidden_sizes, output_dim)

    batch_size: int = 2
    x = npg.randn((batch_size, input_dim), requires_grad=True)
    out = net(x).sum()
    print(out)
    out.backward()
    for param in net.parameters():
        print(param.grad)

    # out = net(x).sum()
    # out.backward()
    # print(net.weight.grad)


if __name__ == "__main__":
    main()
