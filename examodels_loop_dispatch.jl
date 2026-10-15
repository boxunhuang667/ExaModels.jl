# Evaluates an ExaModel's blocks in a loop over a cached `Vector{Any}` instead of by recursion
# over the block tuple. Past 32 blocks that recursion is dynamically dispatched and copies the
# remaining blocks at every level; the loop pays one dynamic call per block. Results are
# unchanged; the measurements are in docs/GPU_performance.md 7.2.
#
# Load after `using ExaModels, KernelAbstractions`. Works on CPU and on a KernelAbstractions GPU.

using KernelAbstractions

const EKA = Base.get_extension(ExaModels, :ExaModelsKernelAbstractions)
const BLOCK_CACHE = IdDict{Any, Vector{Any}}()

# Keyed on `counters`, the model's one mutable field, so a rebuilt model gets its own entry.
block_list(m) = get!(() -> Any[b for b in m.cons], BLOCK_CACHE, m.counters)

# ---- KernelAbstractions path ----

cons_block!(backend, y, con::ExaModels.Constraint, x, θ) =
    isempty(con.itr) || EKA.kerf(backend)(y, con.f, con.itr, x, θ; ndrange = length(con.itr))
cons_block!(backend, y, con, x, θ) = nothing

conaug_block!(backend, y, con::ExaModels.ConstraintAugmentation, x, θ) =
    isempty(con.itr) || EKA.kerf2(backend)(y, con.f, con.itr, x, θ, con.oa; ndrange = length(con.itr))
conaug_block!(backend, y, con, x, θ) = nothing

function ExaModels.cons_nln!(m::ExaModels.AbstractExaModel{T, VT, E}, x::AbstractVector,
        y::AbstractVector) where {T, VT, E <: EKA.KAExtension}
    blocks = block_list(m)
    backend = m.ext.backend
    θ = m.θ
    for i in length(blocks):-1:1
        cons_block!(backend, y, blocks[i], x, θ)
    end
    for i in length(blocks):-1:1
        conaug_block!(backend, m.ext.conbuffer, blocks[i], x, θ)
    end
    if length(m.ext.conaugptr) > 1
        EKA.compress_to_dense(backend)(y, m.ext.conbuffer, m.ext.conaugptr, m.ext.conaugsparsity;
            ndrange = length(m.ext.conaugptr) - 1)
    end
    return y
end

function ExaModels.jac_coord!(m::ExaModels.AbstractExaModel{T, VT, E}, x::V,
        y::V) where {T, VT, E <: EKA.KAExtension, V <: AbstractVector}
    fill!(y, zero(eltype(y)))
    blocks = block_list(m)
    backend = m.ext.backend
    θ = m.θ
    for i in length(blocks):-1:1
        ExaModels.sjacobian!(backend, y, nothing, blocks[i], x, θ, one(eltype(y)))
    end
    return y
end

function ExaModels.hess_coord!(m::ExaModels.AbstractExaModel{T, VT, E}, x::AbstractVector,
        y::AbstractVector, hess::AbstractVector; obj_weight = one(T)) where {T, VT, E <: EKA.KAExtension}
    fill!(hess, zero(eltype(hess)))
    backend = m.ext.backend
    θ = m.θ
    EKA._obj_hess_coord!(backend, hess, m.objs, x, θ, T(obj_weight))
    blocks = block_list(m)
    for i in length(blocks):-1:1
        ExaModels.shessian!(backend, hess, nothing, blocks[i], x, θ, y, zero(eltype(hess)))
    end
    return hess
end

# ---- plain CPU path (backend = nothing) ----

function ExaModels.cons_nln!(m::ExaModels.ExaModel{T, VT, Nothing}, x::AbstractVector,
        g::AbstractVector) where {T, VT}
    fill!(g, zero(eltype(g)))
    blocks = block_list(m)
    θ = m.θ
    for i in length(blocks):-1:1
        ExaModels._cons_rows!(g, blocks[i], x, θ)
    end
    return g
end

function ExaModels.jac_coord!(m::ExaModels.ExaModel{T, VT, Nothing}, x::AbstractVector,
        jac::AbstractVector) where {T, VT}
    fill!(jac, zero(eltype(jac)))
    blocks = block_list(m)
    θ = m.θ
    for i in length(blocks):-1:1
        ExaModels.sjacobian!(jac, nothing, blocks[i], x, θ, one(eltype(jac)))
    end
    return jac
end

function ExaModels.hess_coord!(m::ExaModels.ExaModel{T, VT, Nothing}, x::AbstractVector,
        y::AbstractVector, hess::AbstractVector; obj_weight = one(eltype(x))) where {T, VT}
    fill!(hess, zero(eltype(hess)))
    θ = m.θ
    ExaModels._obj_hess_coord!(m.objs, x, θ, hess, obj_weight)
    blocks = block_list(m)
    for i in length(blocks):-1:1
        ExaModels.shessian!(hess, nothing, blocks[i], x, θ, y, zero(eltype(hess)))
    end
    return hess
end
