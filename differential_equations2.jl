#=

Contains all the RK4 methods for little_boxes_multi_RK


=#

function f_η_0(Ω::Float64, ξ_0_old::ComplexF64)

    return (-Ω/2) * ξ_0_old

end

function f_ξ_0(λL::ComplexF64, η_1_old::Array{ComplexF64}, λR::ComplexF64, η_0_old::ComplexF64)

    return (λL * η_1_old[1]) + (λR * η_1_old[N]) + (Ω/2)*η_0_old

end

function f_η_11(λL::ComplexF64, ξ_0_old::ComplexF64, Ω::Float64, ξ_1_old::Array{ComplexF64})

    return (λL * ξ_0_old) - (Ω/2)*ξ_1_old[1]

end

function f_η_1N(λR::ComplexF64, ξ_0_old::ComplexF64, Ω::Float64, ξ_1_old::Array{ComplexF64})

    return (λR * ξ_0_old) - (Ω/2)*ξ_1_old[N]

end

function f_η_1(j::Int64, Ω::Float64, ξ_1_old::Array{ComplexF64})

    return (-Ω/2) * ξ_1_old[j]

end

function f_ξ_11(λR::ComplexF64, η_2_old::Array{ComplexF64}, Ω::Float64, η_1_old::Array{ComplexF64})

    return (λR * η_2_old[1,1]) + (Ω/2)*η_1_old[1]

end

function f_ξ_1N(λL::ComplexF64, η_2_old::Array{ComplexF64}, Ω::Float64, η_1_old::Array{ComplexF64})

    return (λL * η_2_old[1,N]) + (Ω/2)*η_1_old[N]

end

function f_ξ_1(j::Int64, λL::ComplexF64, η_2_old::Array{ComplexF64}, λR::ComplexF64 , Ω::Float64, η_1_old::Array{ComplexF64})

    return (λL*η_2_old[j,1]) + (λR*η_2_old[j,N]) + (Ω/2)*η_1_old[j]

end

function f_η_2j1(j::Int64, λL::ComplexF64, ξ_1_old::Array{ComplexF64}, Ω::Float64, ξ_2_old::Array{ComplexF64})

    return (λL*ξ_1_old[j]) - (Ω/2)*ξ_2_old[j,1]

end

function f_η_2jN(j::Int64, λR::ComplexF64, ξ_1_old::Array{ComplexF64}, Ω::Float64, ξ_2_old::Array{ComplexF64})

    return (λR*ξ_1_old[j]) - (Ω/2)*ξ_2_old[j,N]

end

function f_η_2(j::Int64, Ω::Float64, ξ_2_old::Array{ComplexF64}, k::Int64)

    return (-Ω/2)*ξ_2_old[j,k]

end

function f_ξ_2(Ω::Float64, η_2_old::Array{ComplexF64}, j::Int64, k::Int64)

    return (Ω/2)*η_2_old[j,k]

end