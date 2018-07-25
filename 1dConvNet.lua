require 'nn'
require 'torch'
require 'math'
require 'gnuplot'

print('<------------Block2 simulation--------->')

print('***1dCovnet, part-1:')
inp=9;  -- dimensionality of one sequence element
outp=20; -- number of derived features for one sequence element
kw=3;   -- kernel only operates on one sequence element per step
dw=1;   -- we step once and go on to the next sequence element

block2_cov1=nn.TemporalConvolution(inp,outp,kw,dw)

imgdata =torch.rand(22,inp) -- a sequence of 7 elements

print('Random spectial data')
print(imgdata)
print('Weight:')
print(block2_cov1.weight)
print('Bias')
print(block2_cov1.bias)

post_cov1 = block2_cov1:forward(imgdata)
print('dimensionality after first 1d')
print(post_cov1)



print('*********1dCovnet, part-2:************\n')
inp=20;  -- dimensionality of one sequence element
outp=10; -- number of derived features for one sequence element
kw=3;   -- kernel only operates on one sequence element per step
dw=1;   -- we step once and go on to the next sequence element

block2_cov2=nn.TemporalConvolution(inp,outp,kw,dw)

print('Weight:')
print(block2_cov2.weight)
print('Bias')
print(block2_cov2.bias)
post_cov2 = block2_cov2:forward(post_cov1)
print(post_cov2)


print('*********1dCovnet, part-3:************\n')
inp=10;  -- dimensionality of one sequence element
outp=5; -- number of derived features for one sequence element
kw=3;   -- kernel only operates on one sequence element per step
dw=1;   -- we step once and go on to the next sequence element

block2_cov3=nn.TemporalConvolution(inp,outp,kw,dw)
print('Weight:')
print(block2_cov3.weight)
print('Bias')
print(block2_cov3.bias)
post_cov3 = block2_cov3:forward(post_cov2)
print(post_cov3)

print('*********1dCovnet, part-4:************\n')
inp=5;  -- dimensionality of one sequence element
outp=5; -- number of derived features for one sequence element
kw=5;   -- kernel only operates on one sequence element per step
dw=1;   -- we step once and go on to the next sequence element

block2_cov4=nn.TemporalConvolution(inp,outp,kw,dw)
print('Weight:')
print(block2_cov4.weight)
print('Bias')
print(block2_cov4.bias)
post_cov4 = block2_cov4:forward(post_cov3)
print(post_cov4)