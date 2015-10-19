
include("TriPMatrices.jl")
n = 6
D = [linspace(1,n,n);] + 5.0
Dl = [linspace(1,n,n);] + 1.0
Du = [linspace(1,n,n);] + 2.0

M = TriPMatrices.TridiagonalP(Dl, D, Du)
Mf = full(M)


S = TriPMatrices.SymTridiagonalP(D, Du)
Sf = full(S)
