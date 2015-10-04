
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

import Base.size

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

import Base.full
full{T}(M::TridiagonalP{T}) = convert(Matrix{T}, M)

import Base.convert
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

import Base.similar
function similar(M::TridiagonalP, T, dims::Dims)
    if length(dims) != 2 || dims[1] != dims[2]
        throw(DimensionMismatch("Periodic Tridiagonal matrices must be square"))
    end
    TridiagonalP{T}(simiular(M.dl), similar(M.d), similar(M.du), similar(M.du2), similar(M.x2))
end

import Base.copy!
copy!(dest::TridiagonalP, src::TridiagonalP) = TridiagonalP(copy!(dest.dl, src.dl), copy!(dest.d, src.d), copy!(dest.du, src.du), copy!(dest.du2, src.du2), copy!(dest.x2, src.x2))



#Elementary operations
import Base.conj, Base.copy, Base.round, Base.trunc, Base.floor, Base.ceil, Base.abs, Base.real, Base.imag

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

import Base.transpose, Base.ctranspose
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


import Base.diag

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


import Base.getindex
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

import  Base.istriu, Base.istril
istriu(M::TridiagonalP) = all(M.dl[2:end] .== 0) && M.du[end] == 0
istriu(M::TridiagonalP) = all(M.du[1:end-1] .== 0) && M.dl[1] == 0


import Base.tril!, Base.triu!

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