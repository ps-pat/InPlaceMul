module InplaceMul

export mulB!

using LinearAlgebra: lmul!

"""
    mulB!(A, B)

For a square matrix `A`, compute `A * B` quasi in place, storing the result
in `B`.
"""
function mulB!(A::Matrix, B::Matrix)
    nrowA = size(A, 1)

    @inbounds for colB ∈ range(1, length = size(B, 2))
        tmp = B[:,colB]
        B[:,colB] .= zero(eltype(B))

        @inbounds @simd for k ∈ eachindex(A)
            B[(k - 1) % nrowA + 1, colB] +=
                tmp[(k - 1) ÷ nrowA + 1] * A[k]
        end
    end

    B
end

end
