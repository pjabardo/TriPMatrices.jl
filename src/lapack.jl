const liblapack = Base.liblapack_name

import Base.blasfunc

import Base.LinAlg: BlasFloat, Char, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, chksquare

#Generic LAPACK error handlers
macro assertargsok() #Handle only negative info codes - use only if positive info code is useful!
    :(info[1]<0 && throw(ArgumentError("invalid argument #$(-info[1]) to LAPACK call")))
end
macro lapackerror() #Handle all nonzero info codes
    :(info>0 ? throw(LAPACKException(info[1])) : @assertargsok )
end

macro assertnonsingular()
    :(info[1]>0 && throw(SingularException(info[1])))
end
macro assertposdef()
    :(info[1]>0 && throw(PosDefException(info[1])))
end

#Check that upper/lower (for special matrices) is correctly specified
function chkuplo(uplo::Char)
    (uplo=='U' || uplo=='L') ||
      throw(ArgumentError(string("uplo argument must be 'U' (upper) or 'L' (lower), got $uplo")))
end

#Check that {c}transpose is correctly specified
function chktrans(trans::Char)
    (trans=='N' || trans=='C' || trans=='T') ||
      throw(ArgumentError(string("trans argument must be 'N' (no transpose), 'T' (transpose), or 'C' (conjugate transpose), got $trans")))
end

#Check that left/right hand side multiply is correctly specified
function chkside(side::Char)
    (side=='L' || side=='R') ||
      throw(ArgumentError(string("side argument must be 'L' (left hand multiply) or 'R' (right hand multiply), got $side")))
end

#Check that unit diagonal flag is correctly specified
function chkdiag(diag::Char)
    (diag=='U' || diag=='N') ||
      throw(ArgumentError(string("diag argument must be 'U' (unit diagonal) or 'N' (non-unit diagonal), got $diag")))
end


# (GT) General tridiagonal, decomposition, solver and direct solver
for (gttrf, gttrs, elty) in
    ((:dgttrf_,:dgttrs_,:Float64),
     (:sgttrf_,:sgttrs_,:Float32),
     (:zgttrf_,:zgttrs_,:Complex128),
     (:cgttrf_,:cgttrs_,:Complex64))
    @eval begin
        #       SUBROUTINE DGTTRF( N, DL, D, DU, DU2, IPIV, INFO )
        #       .. Scalar Arguments ..
        #       INTEGER            INFO, N
        #       .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   D( * ), DL( * ), DU( * ), DU2( * )
        function gttrf!(dl::StridedVector{$elty}, d::StridedVector{$elty}, du::StridedVector{$elty},
                        du2::StridedVector{$elty}, ipiv::StridedVector{BlasInt})
            n    = length(d)
            if length(dl) != n - 1
                throw(DimensionMismatch("Subdiagonal has length $(length(dl)), but should be $(n - 1)"))
            end
            if length(du) != n - 1
                throw(DimensionMismatch("Superdiagonal has length $(length(du)), but should be $(n - 1)"))
            end
            #du2  = similar(d, $elty, n-2)
            #ipiv = similar(d, BlasInt, n)
            info = convert(BlasInt, 0)
            ccall(($(blasfunc(gttrf)), liblapack), Void,
                  (Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                   Ptr{BlasInt}, Ptr{BlasInt}),
                  &n, dl, d, du, du2, ipiv, &info)
            @lapackerror
            dl, d, du, du2, ipiv
        end
        #       SUBROUTINE DGTTRS( TRANS, N, NRHS, DL, D, DU, DU2, IPIV, B, LDB, INFO )
        #       .. Scalar Arguments ..
        #       CHARACTER          TRANS
        #       INTEGER            INFO, LDB, N, NRHS
        #       .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   B( LDB, * ), D( * ), DL( * ), DU( * ), DU2( * )
        function gttrs!(trans::Char, dl::StridedVector{$elty}, d::StridedVector{$elty},
                        du::StridedVector{$elty}, du2::StridedVector{$elty},
                        ipiv::StridedVector{BlasInt}, B::StridedVecOrMat{$elty})
            chktrans(trans)
            chkstride1(B)
            n = length(d)
            if length(dl) != n - 1
                throw(DimensionMismatch("Subdiagonal has length $(length(dl)), but should be $(n - 1)"))
            end
            if length(du) != n - 1
                throw(DimensionMismatch("Superdiagonal has length $(length(du)), but should be $(n - 1)"))
            end
            if n != size(B,1)
                throw(DimensionMismatch("B has leading dimension $(size(B,1)), but should have $n"))
            end
            info = convert(BlasInt, 0)
            ccall(($(blasfunc(gttrs)), liblapack), Void,
                   (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                    Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                    Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                   &trans, &n, &size(B,2), dl, d, du, du2, ipiv, B, &max(1,stride(B,2)), &info)
             @lapackerror
             B
         end
    end
end

