import numpygrad as npg

npg.manual_seed(0)

x = npg.random.randn((3, 2))
mask = x < 0
print(x)
print(mask)
