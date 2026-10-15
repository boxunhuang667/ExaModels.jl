# Method additions that let ExaModels' family merge (`merge.jl` on the pinned
# `compile-time-prototypes` branch) build on a KernelAbstractions GPU backend. The package is not
# edited. What each piece fixes, and the CPU parity check: docs/GPU_performance.md 4.2 and 7.1.
#
# Load after `using ExaModels, KernelAbstractions` and before the core is built.

using KernelAbstractions

# Row targets are computed from a host copy of the iterator, never from the device array.
_host_offset0(b::ExaModels.Constraint, itr, r) = ExaModels.offset0(b.f, itr, r, Base.size(itr))
_host_offset0(b::ExaModels.ConstraintAugmentation, itr, r) = ExaModels.offset0(b.f, itr, r, b.dims)

# A merged augmentation family stays an augmentation: its rows share targets, so they must go
# through the constraint buffer and `compress_to_dense` rather than race on the dense row.
_merged_block(b::ExaModels.Constraint, mf, rows, nrows) =
    ExaModels.Constraint(mf, rows, 0, (nrows,), b.tag)
_merged_block(b::ExaModels.ConstraintAugmentation, mf, rows, nrows) =
    ExaModels.ConstraintAugmentation(mf, rows, 0, (nrows,), b.tag)

# The GPU form of `ExaModels._merge_group`: one device array of `MergedRow` per family, no host
# `SegmentedItr`. A family whose blocks are all empty is refused and stays unmerged.
function ExaModels._merge_group(::Type{T}, blocks::Vector{Any}, backend::KernelAbstractions.GPU,
        θext::Vector{T}, θlen0::Int) where {T}
    acc = (i = Vector{Any}[], f = Vector{Any}[])
    tree = ExaModels._hoist_root(Any[b.f.f for b in blocks], acc)
    mf = ExaModels._simdfunction(T, tree, 0, 0, 0)
    (mf.o1step == blocks[1].f.o1step && mf.o2step == blocks[1].f.o2step) ||
        throw(ExaModels._MergeRefuse("o1step/o2step mismatch"))
    all(b -> b.tag === blocks[1].tag, blocks) || throw(ExaModels._MergeRefuse("tags differ"))

    K = length(acc.i)
    S = length(acc.f)
    nrows = sum(b -> length(b.itr), blocks)
    nrows > 0 || throw(ExaModels._MergeRefuse("every block of the family is empty"))
    θadd = T[]

    rows = Vector{ExaModels.MergedRow}(undef, 0)
    for (j, b) in enumerate(blocks)
        iv = ntuple(k -> Int(acc.i[k][j]), K)
        pbase = θlen0 + length(θext) + length(θadd)
        for v in (acc.f[s][j] for s in 1:S)
            push!(θadd, T(v))
        end
        f = b.f
        itr = Array(b.itr)
        for r in eachindex(itr)
            push!(rows, ExaModels.MergedRow(
                _host_offset0(b, itr, r),
                f.o1 + f.o1step * (r - 1),
                f.o2 + f.o2step * (r - 1),
                iv,
                pbase,
                itr[r],
            ))
        end
    end
    rows_c = [r for r in rows]
    isconcretetype(eltype(rows_c)) || throw(ExaModels._MergeRefuse("mixed row types"))
    con = _merged_block(blocks[1], mf, ExaModels.convert_array(rows_c, backend), nrows)
    Base.append!(θext, θadd)
    return con
end

# After the stock merge, every augmentation block gets a fresh contiguous constraint-buffer
# range, because merged families no longer own the ranges their members had.
function ExaModels._merge_families(c::ExaModels.ExaCore{T, VT, B}) where {T, VT, B <: KernelAbstractions.GPU}
    c2 = invoke(ExaModels._merge_families, Tuple{ExaModels.ExaCore}, c)
    cons = Any[b for b in c2.cons]
    oa = 0
    for i in reverse(eachindex(cons))
        b = cons[i]
        if b isa ExaModels.ConstraintAugmentation
            cons[i] = ExaModels.ConstraintAugmentation(b.f, b.itr, oa, b.dims, b.tag)
            oa += length(b.itr)
        end
    end
    return ExaModels.ExaCore(c2; cons = Tuple(cons), nconaug = oa)
end

# `kerf` reads a merged plain family's row through the three-argument `offset0`.
@inline ExaModels.offset0(f::F, itr::AbstractVector{<:ExaModels.MergedRow}, i) where {F <: ExaModels.SIMDFunction} =
    @inbounds getfield(itr[i], :o0)

# The stock kernels take the nonzero offsets from the `SIMDFunction`; a merged row carries its own.
@kernel function kerj_merged(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj))
    I = @index(Global)
    @inbounds begin
        row = itr[I]
        ExaModels.jrpass(
            f(row, ExaModels.AdjointNodeSource(x), θ),
            f.comp1,
            getfield(row, :o0),
            y1,
            y2,
            getfield(row, :o1),
            0,
            adj,
        )
    end
end

@kernel function kerh_merged(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adj1), @Const(adj2))
    I = @index(Global)
    @inbounds begin
        row = itr[I]
        ExaModels.hrpass0(
            f(row, ExaModels.SecondAdjointNodeSource(x), θ),
            f.comp2,
            y1,
            y2,
            getfield(row, :o2),
            0,
            adj1,
            adj2,
        )
    end
end

@kernel function kerh2_merged(y1, y2, @Const(f), @Const(itr), @Const(x), @Const(θ), @Const(adjs1), @Const(adj2))
    I = @index(Global)
    @inbounds begin
        row = itr[I]
        ExaModels.hrpass0(
            f(row, ExaModels.SecondAdjointNodeSource(x), θ),
            f.comp2,
            y1,
            y2,
            getfield(row, :o2),
            0,
            adjs1[getfield(row, :o0)],
            adj2,
        )
    end
end

const MergedItr = AbstractVector{<:ExaModels.MergedRow}

function ExaModels.sjacobian!(backend::B, y1, y2, f::ExaModels.Constraint{F, I}, x, θ, adj) where {B <: KernelAbstractions.Backend, F, I <: MergedItr}
    isempty(f.itr) || kerj_merged(backend)(y1, y2, f.f, f.itr, x, θ, adj; ndrange = length(f.itr))
end
function ExaModels.sjacobian!(backend::B, y1, y2, f::ExaModels.ConstraintAugmentation{F, I}, x, θ, adj) where {B <: KernelAbstractions.Backend, F, I <: MergedItr}
    isempty(f.itr) || kerj_merged(backend)(y1, y2, f.f, f.itr, x, θ, adj; ndrange = length(f.itr))
end

function ExaModels.shessian!(backend::B, y1, y2, f::ExaModels.Constraint{F, I}, x, θ, adj, adj2) where {B <: KernelAbstractions.Backend, F, I <: MergedItr}
    isempty(f.itr) || kerh_merged(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
end
function ExaModels.shessian!(backend::B, y1, y2, f::ExaModels.ConstraintAugmentation{F, I}, x, θ, adj, adj2) where {B <: KernelAbstractions.Backend, F, I <: MergedItr}
    isempty(f.itr) || kerh_merged(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
end
function ExaModels.shessian!(backend::B, y1, y2, f::ExaModels.Constraint{F, I}, x, θ, adj::V, adj2) where {B <: KernelAbstractions.Backend, F, I <: MergedItr, V <: AbstractVector}
    isempty(f.itr) || kerh2_merged(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
end
function ExaModels.shessian!(backend::B, y1, y2, f::ExaModels.ConstraintAugmentation{F, I}, x, θ, adj::V, adj2) where {B <: KernelAbstractions.Backend, F, I <: MergedItr, V <: AbstractVector}
    isempty(f.itr) || kerh2_merged(backend)(y1, y2, f.f, f.itr, x, θ, adj, adj2; ndrange = length(f.itr))
end
