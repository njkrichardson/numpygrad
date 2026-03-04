MNIST
=====

Source: ``examples/mnist/main.py``

Overview
--------

A small convolutional network trained on the MNIST handwritten-digit
dataset. The example shows how to use ``nn.Conv2d`` and how to build a
custom ``Module`` that mixes convolutions with a final linear classifier.

Running
-------

::

    python -m examples.mnist.main          # downloads data on first run
    python -m examples.mnist.main --help   # see all options

Selected options:

- ``--num-steps`` — training steps (default 500)
- ``--batch-size`` — mini-batch size (default 32)
- ``--hidden-dim`` — number of conv channels (default 32)
- ``--step-size`` — AdamW learning rate (default 1e-3)

Code walkthrough
----------------

**Dataset**

MNIST images are downloaded automatically on first run and cached under
``examples/mnist/data/``::

    train_dataset = MNIST(split="train")   # 60 000 images, 28×28 greyscale
    test_dataset  = MNIST(split="test")    # 10 000 images

**Model**

Two convolutional layers followed by a linear output head::

    class MNISTClassifier(nn.Module):
        def __init__(self, input_shape, num_classes, hidden_dim):
            super().__init__()
            self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
            self.linear_out = nn.Linear(hidden_dim * H * W, num_classes)

        def forward(self, x):
            x = npg.relu(self.conv1(x))    # (N, hidden, 28, 28)
            x = npg.relu(self.conv2(x))    # (N, hidden, 28, 28)
            x = x.reshape(x.shape[0], -1)  # (N, hidden*28*28)
            return self.linear_out(x)       # (N, 10)

**Training loop**

::

    optimizer = npg.optim.AdamW(net.parameters(), lr=1e-3)
    for step in range(num_steps):
        x, y = next(iter(dataloader))
        optimizer.zero_grad()
        loss = nn.cross_entropy_loss(net(x), y)
        loss.backward()
        optimizer.step()
