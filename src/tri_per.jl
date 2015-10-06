immutable SymTridiagonalP{T} <: AbstractMatrix{T}
    d::Vector{T}
    du::Vector{T}
    x2::Vector{T}
end

function SymTridiagonalP{T}(d::Vector{T}, du::Vector{T})
    if length(du) != length(d)
        throw(DimensionMismatch("Subdiagonal has wrong length."))
    end
    SymTridiagonalP(d, du, zeros(T, length(d)))
end

import Base.size


size(M::SymTridiagonalP) = (length(M.d), length(M.d))
function size(M::SymTridiagonalP, d::Integer)
    if d < 1
        throw(ArgumentError("dimension d must be ≥ 1, got $d"))
    elseif d <= 2
        return length(M.d)
    else
        return 1
    end
end

import Base.full
full{T}(M::SymTridiagonalP{T}) = convert(Matrix{T}, M)

import Base.convert
function convert{T}(::Type{Matrix{T}}, M::SymTridiagonalP{T})
    A = zeros(T, size(M))
    for i = 1:length(M.d)
        A[i,i] = M.d[i]
    end

    for i = 1:length(M.d)-1
        A[i+1,i] = M.du[i]
        A[i,i+1] = M.du[i]
    end
    A[1, length(M.d)] = M.du[end]
    A[length(M.d), 1] = M.du[end]
    A
end

convert{T}(::Type{Matrix}, M::SymTridiagonalP{T}) = convert(Matrix{T}, M)

import Base.similar
function similar(M::SymTridiagonalP, T, dims::Dims)
    if length(dims) != 2 || dims[1] != dims[2]
        throw(DimensionMismatch("Periodic Tridiagonal matrices must be square"))
    end
    SymTridiagonalP{T}(simiular(M.d), similar(M.du), similar(M.x2))
end

import Base.copy!
copy!(dest::SymTridiagonalP, src::SymTridiagonalP) = SymTridiagonalP(copy!(dest.d, src.d), copy!(dest.du, src.du), copy!(dest.x2, src.x2))



#Elementary operations
import Base.conj, Base.copy, Base.round, Base.trunc, Base.floor, Base.ceil, Base.abs, Base.real, Base.imag

for func in (:conj, :copy, :round, :trunc, :floor, :ceil, :abs, :real, :imag)
    @eval function ($func)(M::SymTridiagonalP)
        SymTridiagonalP(($func)(M.d), ($func)(M.du), ($func)(M.x2))
    end
end


for func in (:round, :trunc, :floor, :ceil)
    @eval function ($func){T<:Integer}(::Type{T},M::SymTridiagonalP)
        SymTridiagonalP(($func)(T,M.d), ($func)(T,M.du), ($func)(T,M.x2))
    end
end

import Base.transpose, Base.ctranspose
transpose(M::SymTridiagonalP) = M
ctranspose(M::SymTridiagonalP) = conj(transpose(M))


import Base.diag

function diag{T}(M::SymTridiagonalP{T}, n::Integer=0)
    n = (n < 0)?-n:n
    if n == 0
        return M.d
    elseif n == 1
        return M.du[1:end-1]
    elseif n == length(M.d)-1
        return [M.du[end]]
    elseif n < size(M,1)
        return zeros(T,size(M,1) - n)
    else
        throw(BoundsError("$n-th diagonal of a $(size(M)) matrix doesn't exist!"))
    end
end


import Base.getindex
function getindex{T}(A::SymTridiagonalP{T}, i::Integer, j::Integer)
    if !(1 <= i <= size(A,2) && 1 <= j <= size(A,2))
        throw(BoundsError("(i,j) = ($i,$j) not within matrix of size $(size(A))"))
    end
    if i == j
        return A.d[i]
    elseif i == j + 1
        return A.du[j]
    elseif i + 1 == j
        return A.du[i]
    elseif i == 1 && j == length(A.d)
        return A.du[end]
    elseif j == 1 && i == length(A.d)
        return A.du[end]
    else
        return zero(T)
    end
end

import  Base.istriu, Base.istril
istriu(M::SymTridiagonalP) = all(M.du .== 0) 
istriu(M::SymTridiagonalP) = all(M.du .== 0)


import Base.tril!, Base.triu!

function tril!(M::SymTridiagonalP, k::Integer=0)
    n = length(M.d)
    cn = M.du[end]
    if abs(k) > n
        throw(ArgumentError("requested diagonal, $k, out of bounds in matrix of size ($n,$n)"))
    elseif k < -1
        fill!(M.du,0)
        dl = copy(M.du)
        if k > -n
            M.du[end] = cn
        end
        fill!(M.d,0)
        return TridiagonalP(dl,M.d,M.du)
    elseif k == -1
        fill!(M.d,0)
        dl = similar(M.du)
        for i = 2:n
            dl[i] = M.du[i-1]
            M.du[i-1] = 0
        end
        dl[1] = 0
        return TridiagonalP(dl,M.d,M.du)
    elseif k == 0
        dl = similar(M.du)
        for i = 2:n
            dl[i] = M.du[i-1]
            M.du[i-1] = 0
        end
        dl[1] = 0
        return TridiagonalP(dl,M.d,M.du)
    elseif k >= 1
        dl = similar(M.du)
        for i = 2:n
            dl[i] = M.du[i-1]
        end
        if k < n-1
            dl[1] = 0
        else
            dl[1] = cn
        end
        return TridiagonalP(dl,M.d,M.du)
    end
end

function triu!(M::SymTridiagonalP, k::Integer=0)
    n = length(M.d)
    cn = M.du[end]
    if abs(k) > n
        throw(ArgumentError("requested diagonal, $k, out of bounds in matrix of size ($n,$n)"))
    elseif k > 1
        
        fill!(M.d,0)
        fill!(M.du,0)
        dl = copy(M.d)
        if k < n
            dl[1] = cn
        end
        M.du[end] = 0
        return TridiagonalP(dl,M.d, M.du)
    elseif k == 1
        fill!(M.d,0)
        dl = zeros(M.du)
        dl[1] = cn
        M.du[end] = 0
        return TridiagonalP(dl, M.d, M.du)
    elseif k == 0
        dl = zeros(M.du)
        dl[1] = cn
        M.du[end] = 0
        return TridiagonalP(dl,M.d,M.du)
    elseif k <= -1
        dl = similar(M.du)
        for i = 2:n
            dl[i] = M.du[i-1]
        end
        dl[1] = cn
        if k > -(n-1)
            M.du[end] = 0
        end
        return TridiagonalP(dl,M.d,M.du)
    end
end

import Base.+, Base.-, Base.*, Base./, Base.==

+(A::SymTridiagonalP, B::SymTridiagonalP) = SymTridiagonalP(A.d + B.d, A.du + B.du)
-(A::SymTridiagonalP, B::SymTridiagonalP) = SymTridiagonalP(A.d-B.d, A.du-B.du)
*(A::SymTridiagonalP, B::Number) = SymTridiagonalP(A.d*B, A.du*B)
*(B::Number, A::SymTridiagonalP) = A*B
/(A::SymTridiagonalP, B::Number) = SymTridiagonalP(A.d/B, A.du/B)
==(A::SymTridiagonalP, B::SymTridiagonalP) = (A.d==B.d) && (A.du==B.du)
        
import Base.A_mul_B!

function A_mul_B!(C::StridedVecOrMat, S::SymTridiagonalP, B::StridedVecOrMat)
    m, n = size(B, 1), size(B, 2)
    if !(m == size(S, 1) == size(C, 1))
        throw(DimensionMismatch("A has first dimension $(size(S,1)), B has $(size(B,1)), C has $(size(C,1)) but all must match"))
    end
    if n != size(C, 2)
        throw(DimensionMismatch("second dimension of B, $n, doesn't match second dimension of C, $(size(C,2))"))
    end

    d = S.d
    u = S.du
    un = u[end]
    
    @inbounds begin
        for j = 1:n
            b₀, b₊, bn = B[1, j], B[2, j], B[m,j]
            b0 = b₀
            C[1, j] = d[1]*b₀ + u[1]*b₊ + un*bn
            for i = 2:m - 1
                b₋, b₀, b₊ = b₀, b₊, B[i + 1, j]
                C[i, j] = u[i-1]*b₋ + d[i]*b₀ + u[i]*b₊
            end
            C[m, j] = un*b0 + u[m-1]*b₀ + d[m]*b₊
        end
    end

    return C
end
              


immutable TridiagonalP{T} <: AbstractMatrix{T}
    dl::Vector{T}
    d::Vector{T}
    du::Vector{T}
    du2::Vector{T}
    x2::Vector{T}
end

TridiagonalP{T}(n::Integer, ::Type{T}=Float64) = TridiagonalP{T}(zeros(T,n), zeros(T,n), zeros(T,n), zeros(T,n-3), zeros(T,n-1))


function TridiagonalP{T}(dl::Vector{T}, d::Vector{T}, du::Vector{T})
    n = length(d)
    if (length(dl) != n) || (length(du) != n)
        throw(ArgumentError("Cannot make Periodic Tridiagonal matrix from incompatible diagonals: ($(length(dl)), $(length(d)), $(length(du))"))
    end
    TridiagonalP(dl, d, du, zeros(T,n-3), zeros(T,n-1))
end


size(M::TridiagonalP) = (length(M.d), length(M.d))
function size(M::TridiagonalP, d::Integer)
    if d < 1
        throw(ArgumentError("dimension d must be ≥ 1, got $d"))
    elseif d <= 2
        return length(M.d)
    else
        return 1
    end
end

full{T}(M::TridiagonalP{T}) = convert(Matrix{T}, M)

function convert{T}(::Type{Matrix{T}}, M::TridiagonalP{T})
    A = zeros(T, size(M))
    for i = 1:length(M.d)
        A[i,i] = M.d[i]
    end

    for i = 1:length(M.d)-1
        A[i+1,i] = M.dl[i+1]
        A[i,i+1] = M.du[i]
    end
    A[1, length(M.d)] = M.dl[1]
    A[length(M.d), 1] = M.du[length(M.d)]
    A
end

convert{T}(::Type{Matrix}, M::TridiagonalP{T}) = convert(Matrix{T}, M)

function similar(M::TridiagonalP, T, dims::Dims)
    if length(dims) != 2 || dims[1] != dims[2]
        throw(DimensionMismatch("Periodic Tridiagonal matrices must be square"))
    end
    TridiagonalP{T}(simiular(M.dl), similar(M.d), similar(M.du), similar(M.du2), similar(M.x2))
end

copy!(dest::TridiagonalP, src::TridiagonalP) = TridiagonalP(copy!(dest.dl, src.dl), copy!(dest.d, src.d), copy!(dest.du, src.du), copy!(dest.du2, src.du2), copy!(dest.x2, src.x2))



#Elementary operations

for func in (:conj, :copy, :round, :trunc, :floor, :ceil, :abs, :real, :imag)
    @eval function ($func)(M::TridiagonalP)
        TridiagonalP(($func)(M.dl), ($func)(M.d), ($func)(M.du), ($func)(M.du2), ($func)(M.x2))
    end
end


for func in (:round, :trunc, :floor, :ceil)
    @eval function ($func){T<:Integer}(::Type{T},M::TridiagonalP)
        TridiagonalP(($func)(T,M.dl), ($func)(T,M.d), ($func)(T,M.du), ($func)(T,M.du2), 
                     ($func)(T,M.x2))
    end
end

function transpose(M::TridiagonalP)
    n = length(M.d)
    b1 = M.dl[1]
    cn = M.du[n]
    dl = M.dl
    du = M.du
    
    for i = 2:n
        tmp = du[i-1]
        du[i-1] = dl[i]
        dl[i] = tmp
    end
    dl[1] = cn
    du[n] = b1

    TridiagonalP(dl, M.d, du)
end


ctranspose(M::TridiagonalP) = conj(transpose(M))



function diag{T}(M::TridiagonalP{T}, n::Integer=0)
    if n == 0
        return M.d
    elseif n == -1
        return M.dl[2:end]
    elseif n == 1
        return M.du[1:end-1]
    elseif n==length(M.d)-1
        return [M.dl[1]]
    elseif n==-(length(M.d)-1)
        return [M.du[end]]
    elseif abs(n) < size(M,1)
        return zeros(T,size(M,1)-abs(n))
    else
        throw(BoundsError("$n-th diagonal of a $(size(M)) matrix doesn't exist!"))
    end
end


function getindex{T}(A::TridiagonalP{T}, i::Integer, j::Integer)
    if !(1 <= i <= size(A,2) && 1 <= j <= size(A,2))
        throw(BoundsError("(i,j) = ($i,$j) not within matrix of size $(size(A))"))
    end
    if i == j
        return A.d[i]
    elseif i == j + 1
        return A.dl[i]
    elseif i + 1 == j
        return A.du[i]
    elseif i == 1 && j == length(A.d)
        return A.dl[1]
    elseif j == 1 && i == length(A.d)
        return A.du[end]
    else
        return zero(T)
    end
end

istriu(M::TridiagonalP) = all(M.dl[2:end] .== 0) && M.du[end] == 0
istriu(M::TridiagonalP) = all(M.du[1:end-1] .== 0) && M.dl[1] == 0



function tril!(M::TridiagonalP, k::Integer=0)
    n = length(M.d)

    cn = M.du[end]
    b1 = M.dl[1]
    M.dl[1] = 0
    
    if abs(k) > n
        throw(ArgumentError("requested diagonal, $k, out of bounds in matrix of size ($n,$n)"))
    elseif k < -1
        fill!(M.dl,0)
        fill!(M.d,0)
        fill!(M.du,0)
    elseif k == -1
        fill!(M.d,0)
        fill!(M.du,0)
    elseif k == 0
        fill!(M.du,0)
    elseif k >= n-1
        M.dl[1] = b1
    end
    if k > -n
        M.du[end] = cn
    end
    return M
end
    
function triu!(M::TridiagonalP, k::Integer=0)
    n = length(M.d)

    b1 = M.dl[1]
    cn = M.du[end]
    M.du[end] = 0
    
    if abs(k) > n
        throw(ArgumentError("requested diagonal, $k, out of bounds in matrix of size ($n,$n)"))
    elseif k > 1
        fill!(M.dl,0)
        fill!(M.d,0)
        fill!(M.du,0)
    elseif k == 1
        fill!(M.dl,0)
        fill!(M.d,0)
    elseif k == 0
        fill!(M.dl,0)
    elseif k <= -(n-1)
        M.du[end] = cn
    end
    if k < n
        M.dl[1] = b1
    end
    return M
end





###################
# Generic methods #
###################

+(A::TridiagonalP, B::TridiagonalP) = TridiagonalP(A.dl+B.dl, A.d+B.d, A.du+B.du)
-(A::TridiagonalP, B::TridiagonalP) = TridiagonalP(A.dl-B.dl, A.d-B.d, A.du-B.du)
*(A::TridiagonalP, B::Number) = TridiagonalP(A.dl*B, A.d*B, A.du*B)
*(B::Number, A::TridiagonalP) = A*B
/(A::TridiagonalP, B::Number) = TridiagonalP(A.dl/B, A.d/B, A.du/B)


function A_mul_B!(C::AbstractVecOrMat, A::TridiagonalP, B::AbstractVecOrMat)
    nA = size(A,1)
    nB = size(B,2)
    if !(size(C,1) == size(B,1) == nA)
        throw(DimensionMismatch("A has first dimension $nA, B has $(size(B,1)), C has $(size(C,1)) but all must match"))
    end
    
    if size(C,2) != nB
        throw(DimensionMismatch("A has second dimension $nA, B has $(size(B,2)), C has $(size(C,2)) but all must match"))
    end
    l = A.dl
    d = A.d
    u = A.du
    un = A.du[end]
    l₁ = A.dl[1]
    @inbounds begin
        for j = 1:nB
            b₀, b₊, bn = B[1, j], B[2, j], B[nA,j]
            b0 = b₀
            C[1, j] = d[1]*b₀ + u[1]*b₊ + l₁*bn
            for i = 2:nA - 1
                b₋, b₀, b₊ = b₀, b₊, B[i + 1, j]
                C[i, j] = l[i]*b₋ + d[i]*b₀ + u[i]*b₊
            end
            C[nA, j] = un*b0 + l[nA]*b₀ + d[nA]*b₊
        end
    end
    C
end
