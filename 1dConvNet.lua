require 'nn'
require 'torch'
require 'math'

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



print('\n--------------------Custom Test 1-------------\n')


-- 7x3 -> 7x1
x = torch.Tensor(7,3)
s = x:storage()
for i=1,s:size() do -- fill up the Storage
  s[i] = i
end


weight = torch.Tensor(1,3)
s = weight:storage()
for i=1,s:size() do -- fill up the Storage
    s[i] = 3
end


print('Fake weight')
print(weight)

print('Fake data')
print(x)

print('result')
weights = torch.reshape(weight, 3)
for index=1, x:size(1) do
    ele = x[index];
    dot = ele:dot(weights)
    print(dot)
end


print('\n--------------------Custom Test 2-------------\n')

-- Input 7x3 -> 5x6 , kenal = 3

x = torch.Tensor(7,3)
s = x:storage()
for i=1,s:size() do -- fill up the Storage
  s[i] = i
end

print('Fake Data')
print(x)

weight = torch.Tensor(6,3)
s = weight:storage()
for i=1,s:size() do -- fill up the Storage
    s[i] = 3
end


print('Fake weight')
print(weight)

fakeModel = nn.TemporalConvolution(3,6,3,1)
print(fakeModel:forward(x))
print(fakeModel.weight)
--for index=1, x:size(1) do
--    ele = x[index];

--    print(ele)
--end