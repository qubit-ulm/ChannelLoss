module qecc

using PolynomialOptimization
using MultivariatePolynomials
using DynamicPolynomials
using LinearAlgebra
using Combinatorics
import Combinatorics
import TensorOperations

const newmp = isdefined(MultivariatePolynomials, :polynomial_type)

#region https://github.com/Jutho/VectorInterface.jl/pull/4/files
import VectorInterface

const PolyTypes = Union{<:AbstractPolynomialLike, <:AbstractTermLike, <:AbstractMonomialLike}

# not clear if this is really the true `scalartype` we want
VectorInterface.scalartype(T::Type{<:PolyTypes}) = T

function VectorInterface.add!!(w::PolyTypes, v::PolyTypes, α::Number, β::Number)
    return w * β + v * α
end

VectorInterface.scale!!(v::PolyTypes, α::Number) = v * α
#endregion

const circ = Combinatorics.multiexponents;
multi(n::Integer, vals::Vector{<:Integer}) = all(vals .≥ 0) && sum(vals) == n ? Combinatorics.multinomial(vals...) : 0;

function dTens(d::Integer, s::Integer, r::Integer)
    @assert d >= 2 && s >= r >= 1
    sdIt = circ(d, s)
    rdIt = circ(d, r)
    return reshape(Float64[(all(kp .- k .== lp .- l) ? multi(s - r, kp .- k) * sqrt((multi(r, k) * multi(r, l)) /
                                                                                    (multi(s, kp) * multi(s, lp)))
                            : 0.0)
                           for lp in sdIt for kp in sdIt for l in rdIt for k in rdIt],
        (length(rdIt), length(rdIt), length(sdIt), length(sdIt)))
end;

function symmetricPolyMatrix(name::AbstractString, size::Integer)
    vars = Array{newmp ? Variable{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder},Graded{LexOrder}} :
                         DynamicPolynomials.PolyVar{true}}(
        undef, (size * (size + 1)) >> 1
    )
    mat = Matrix{(newmp ? polynomial_type : polynomialtype)(eltype(vars),Float64)}(undef, size, size)
    idx = 1
    for j in 1:size # we need to access both row-wise and col-wise, so order doesn't matter
        for i in 1:j
            var = newmp ? Variable("$name[$i, $j]", DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder},
                                   Graded{LexOrder}) :
                          DynamicPolynomials.PolyVar{true}("$name[$i, $j]")
            @inbounds mat[i, j] = mat[j, i] = vars[idx] = var
            idx += 1
        end
    end
    return mat, vars
end;

function hermitianPolyMatrix(name::AbstractString, size::Integer)
    vars = Array{newmp ? Variable{DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder},Graded{LexOrder}} :
                         DynamicPolynomials.PolyVar{true}}(
        undef, (size * (size + 1)) >> 1
    )
    mat = Matrix{(newmp ? polynomial_type : polynomialtype)(eltype(vars),ComplexF64)}(undef, size, size)
    idx = 1
    for j in 1:size # we need to access both row-wise and col-wise, so order doesn't matter
        for i in 1:j-1
            var = newmp ? Variable("$name[$i, $j]", DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}, Graded{LexOrder},
                                   DynamicPolynomials.COMPLEX) :
                          DynamicPolynomials.PolyVar{true}("$name[$i, $j]", Val(:complex))
            @inbounds mat[i, j] = vars[idx] = var
            @inbounds mat[j, i] = conj(var)
            idx += 1
        end
        var = newmp ? Variable("$name[$j, $j]", DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder},
                               Graded{LexOrder}) :
                      DynamicPolynomials.PolyVar{true}("$name[$j, $j]")
        @inbounds mat[j, j] = vars[idx] = var
        idx += 1
    end
    return mat, vars
end;

function optProgRed(d::Integer, s::Integer, r::Integer, pdist::Float64; complex::Bool=false)
    @assert d >= 2 && s >= r >= 1 && 0 <= pdist <= 1
    fn = complex ? hermitianPolyMatrix : symmetricPolyMatrix
    ρSize = 2binomial(s + d - 1, d - 1)
    ρ, ρVars = fn("ρ", ρSize)
    ρT = reshape(ρ, (ρSize >> 1, 2, ρSize >> 1, 2))
    choiSize = 2binomial(r + d - 1, d - 1)
    choi, choiVars = fn("choi", choiSize)
    choiT = reshape(choi, (choiSize >> 1, 2, choiSize >> 1, 2))
    ρFinal = zeros(eltype(ρ), 2, 2, 2, 2)
    dT = dTens(d, s, r)
    TensorOperations.@tensor ρFinal[bobR, aliceR, bobC, aliceC] = choiT[recvR, bobR, recvC, bobC] *
                                                                  dT[recvR, recvC, sendR, sendC] *
                                                                  ρT[sendR, aliceR, sendC, aliceC]
    ρFinal = reshape(ρFinal, (4, 4))
    fidelity = (ρFinal[1, 1] + ρFinal[1, 4] + ρFinal[4, 1] + ρFinal[4, 4]) / (2pdist)
    TensorOperations.@tensor choiPT[inR, inC] := choiT[inR, out, inC, out]
    return poly_problem(-fidelity, zero=[tr(ρFinal) - pdist, 1 - tr(ρ)], psd=[ρ, choi, LinearAlgebra.I - choiPT])
end;

function optProgFull(d::Integer, r::Integer, basis::AbstractMatrix{Bool}, pdist::Float64; complex::Bool=false)
    s = size(basis, 1) -1
    @assert d >= 2 && s >= r >= 1 && 0 <= pdist <= 1
    fn = complex ? hermitianPolyMatrix : symmetricPolyMatrix
    ρ, ρVars = fn("ρ", size(basis, 2))
    PT = polynomial_type(eltype(ρ), complex ? ComplexF64 : Float64)
    radii = [size(ρ, 1)^2 - sum(x -> x^2, ρ)]
    total_trace = zero(PT)
    total_overlap = zero(PT)
    conditions = PT[]
    psd_conditions = Matrix{PT}[]
    nonnegs = PT[]
    push!(psd_conditions, ρ)
    nslots = binomial(s, r)
    # The code commented out allows to use individual normalizations instead of the simplified version. However, note that the
    # second level will be unbouded, which can be mitigated by introducing trivial box constraints.
    #@polyvar overlaps[1:nslots] traces[1:nslots]
    @inbounds for (combi, arrival) in enumerate(combinations(2:s+1, r))
        # println("Combination $combi - arrivals: ", arrival .- 1)
        # ρRedBasis is constructed from the first basis element (Alice) + all the ones indexed in arrival.
        # The basis elements are identified by integers with r +1 bits. First (highest) is Alice, rest are received slots.
        ρRedBasisElems = Set{UInt}()
        for basisᵢ in eachcol(basis)
            i = UInt(basisᵢ[1])
            for arrivalᵢ in arrival
                i <<= 1
                basisᵢ[arrivalᵢ] && (i |= true)
            end
            push!(ρRedBasisElems, i)
        end
        ρRedBasis = sort!(collect(ρRedBasisElems))
        # println("ρ reduced basis: ", string.(ρRedBasis, base=2, pad=r+1))
        # ρRed is constructed by tracing over all lost ones
        lost = setdiff(2:s+1, arrival)
        ρRed = zeros(PT, length(ρRedBasis), length(ρRedBasis))
        for (jj, basisⱼ) in enumerate(eachcol(basis))
            j = UInt(basisⱼ[1])
            for arrivalⱼ in arrival
                j <<= 1
                basisⱼ[arrivalⱼ] && (j |= true)
            end
            jpos = findfirst(isequal(j), ρRedBasis)
            for (ii, basisᵢ) in enumerate(eachcol(basis))
                @views basisⱼ[lost] == basisᵢ[lost] || continue
                i = UInt(basisᵢ[1])
                for arrivalᵢ in arrival
                    i <<= 1
                    basisᵢ[arrivalᵢ] && (i |= true)
                end
                ipos = findfirst(isequal(i), ρRedBasis)
                ρRed[ipos, jpos] += ρ[ii, jj]
            end
        end
        aliceBit = one(UInt) << r
        receiveMask = aliceBit - one(UInt)
        # choiBasis needs all reduced basis elements (received part only) plus 0 and 1 for output as the highest bit (which is
        # the same as where the Alice bit was coded).
        choiBasis = sort!(collect(Set((x & receiveMask) | receiveOut for x in ρRedBasisElems for receiveOut in (zero(UInt), aliceBit))))
        # println("Choi basis: ", string.(choiBasis, base=2, pad=r+1))
        choi, choiVars = fn("choi" * string(combi), length(choiBasis))
        push!(radii, size(choi, 1)^2 - sum(x -> x^2, choi))
        push!(psd_conditions, choi)

        # we must impose the tr_out choi ⪯ 𝟙 condition.
        choiPTBasis = sort!(collect(Set(x & receiveMask for x in choiBasis)))
        choiPT = Matrix{PT}(undef, length(choiPTBasis), length(choiPTBasis))
        choiPT .= LinearAlgebra.I(length(choiPTBasis))
        for pt in (zero(UInt), aliceBit)
            for (jj, basisⱼ) in enumerate(choiPTBasis)
                j = findfirst(isequal(basisⱼ | pt), choiBasis)
                isnothing(j) && continue
                for (ii, basisᵢ) in enumerate(choiPTBasis)
                    i = findfirst(isequal(basisᵢ | pt), choiBasis)
                    isnothing(i) && continue
                    choiPT[ii, jj] -= choi[i, j]
                end
            end
        end
        push!(psd_conditions, choiPT)

        # we are only interested in the overlap with |Φ⁺⟩ in the end, so we don't need to calculate the whole ρFinal. Only the
        # |00⟩⟨00|, |00⟩⟨11|, hc, |11⟩⟨11| components are required.
        # However, we also need the trace, meaning |01⟩⟨01| and |10⟩⟨10|.
        for (cj, choiBasisⱼ) in enumerate(choiBasis)
            j = findfirst(isequal(choiBasisⱼ), ρRedBasis) # this checks both that Alice's bit matches the output bit (from the
                                                          # overlap and that the input matches the state)
            if !isnothing(j)
                for (ci, choiBasisᵢ) in enumerate(choiBasis)
                    i = findfirst(isequal(choiBasisᵢ), ρRedBasis)
                    isnothing(i) && continue
                    total_overlap += choi[ci, cj] * ρRed[i, j]
                    if choiBasisᵢ & aliceBit == choiBasisⱼ & aliceBit
                        total_trace += choi[ci, cj] * ρRed[i, j]
                    end
                end
            end
            # non-matching outputs
            choiBasisⱼ ⊻= aliceBit
            j = findfirst(isequal(choiBasisⱼ), ρRedBasis)
            isnothing(j) && continue
            for (ci, choiBasisᵢ) in enumerate(choiBasis)
                choiBasisᵢ ⊻= aliceBit
                choiBasisᵢ & aliceBit == choiBasisⱼ & aliceBit || continue
                i = findfirst(isequal(choiBasisᵢ), ρRedBasis)
                isnothing(i) && continue
                total_trace += choi[ci, cj] * ρRed[i, j]
            end
        end
        # push!(conditions, 2 * overlaps[combi] - overlap, traces[combi] - trace,
        #     traces[combi] * fidelities[combi] - overlaps[combi])
        # push!(nonnegs, overlaps[combi], traces[combi], fidelities[combi], 1 - traces[combi], 1 - fidelities[combi],
        #     traces[combi] - overlaps[combi])
        # total_trace += trace
    end
    total_overlap /= 2nslots
    total_trace /= nslots
    fidelity = total_overlap / pdist
    # fidelity = sum(overlap) / nslots
    return poly_problem(-fidelity, zero=push!(conditions, total_trace - pdist, 1 - tr(ρ)),
        psd=psd_conditions, nonneg=nonnegs)
end

end

import .qecc
using PolynomialOptimization, Mosek, SCS
# If you want to use the bases generated by the Python script:
#=
using PythonCall
sys = pyimport("sys")
sys.path.append(@__DIR__)
gs = pyimport("GenerateStepwise")
for pattern in gs.getPatterns(d, s, r, 2)
    prob = qecc.optProgFull(d, r, [Bool(pattern[i][j]) for j in 0:s, i in 0:length(pattern)-1], pdist)
    rel = Relaxation.Newton(prob, 2)
    println(pattern, "\t", poly_optimize(:SCS, rel).objective)
    flush(stdout)
end
println("Done")=#

# If you want to use the reduced formulation:
#=
prob = qecc.optProgRed(d, s, r, pdist)
rel = Relaxation.Newton(prob, 2)
poly_optimize(:MosekSOS, rel)
=#