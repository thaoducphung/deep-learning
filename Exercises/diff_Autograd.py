import torch 
a = torch.tensor([2.,3.],requires_grad=True)
b = torch.tensor([6.,4.],requires_grad=True)

Q = 3*a**3 - b**2

exeternal_grad = torch.tensor([1.,1.])
Q.backward(gradient=exeternal_grad)

print('a.grad',a.grad)

print('b.grad',b.grad)

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

x = torch.rand(5,5)
print(x)

y = torch.rand(5,5)
print(y)

z = torch.rand((5,5),requires_grad=True)
a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

print(b.grad)