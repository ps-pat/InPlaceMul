module InplaceMul

export mulB!

using LinearAlgebra: lmul!

function mulB!(A::Matrix, B::Matrix)
    @inbounds for colB ∈ range(1, stop = size(B, 2))
        tmp = B[:,colB]
        B[:,colB] .= zero(eltype(B))

        @inbounds @simd for colA ∈ range(1, stop = size(A, 2))
            B[:,colB] .+= tmp[colA] * @view(A[:,colA])
        end
    end

    B
end

end
