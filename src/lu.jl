

using ArrayViews

#import Base.LinAlg.LAPACK
import Base.LinAlg.BLAS.BlasFloat
import Base.LinAlg.BLAS.BlasInt
import Base.lufact!
#using Base.LinAlg.LAPACK.gttrs!
using Base.LU

function lufact!{T<:BlasFloat}(A::TridiagonalP{T}, pivot::Union{Type{Val{false}}, Type{Val{true}}} = Val{true})
    n = length(A.d)
    cn = A.du[n]
    b1 = A.dl[1]

    ipiv = zeros(BlasInt, n-1)
    dl1, d, du1, du2, ipiv = Lapack.gttrf!(view(A.dl, 2:n-1), view(A.d, 1:n-1), view(A.du, 1:n-2), A.du2, ipiv)
    LU{T,TridiagonalP{T}}(A, ipiv, 0)
end


import Base.factorize
factorize(A::TridiagonalP) = lufact(A)

import Base.A_ldiv_B!
function A_ldiv_B!{T<:BlasFloat}(A::LU{T,TridiagonalP{T}}, B::AbstractVecOrMat)
    n = size(A, 1)
    n1 = n-1
    qn1 = view(B, 1:n1, :)
    nrhs = size(B,2)

    D = view(A.factors.d, 1:n1)
    Dl = view(A.factors.dl, 2:n1)
    Du = view(A.factors.du, 1:n-2)
    Lapack.gttrs!('N', Dl, D, Du, A.factors.du2, A.ipiv, qn1)
    
    x2 = A.factors.x2
    

    
    b1  = A.factors.dl[1]
    bn = A.factors.dl[n]
    cn1 = A.factors.du[n1]
    cn = A.factors.du[n]
    an = A.factors.d[n]
    
    for i = 1:nrhs
        fill!(x2, 0)
        x2[1] = -b1
        x2[n1] = -cn1
        Lapack.gttrs!('N', Dl, D, Du, A.factors.du2, A.ipiv, x2)
        xn = (B[n,i] - cn*qn1[1,i] - bn*qn1[n1,i]) / (an + cn*x2[1] + bn*x2[n1])
        B[n,i] = xn
        for k = 1:n1
            B[k,i] = B[k,i] + xn*x2[k]
        end
    end
    B
end
