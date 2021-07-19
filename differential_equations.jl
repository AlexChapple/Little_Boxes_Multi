#=

Contains all the RK4 methods for little_boxes_multi_RK


=#

function f_η_0(Ω, ξ_0_old)

    return (-Ω/2) * ξ_0_old

end

function f_ξ_0(λL, η_1_old, λR, η_0_old)

    return (λL * η_1_old[1]) + (λR * η_1_old[N]) + (Ω/2)*η_0_old

end

function f_η_11(λL, ξ_0_old, Ω, ξ_1_old)

    return (λL * ξ_0_old) - (Ω/2)*ξ_1_old[1]

end

function f_η_1N(λR, ξ_0_old, Ω, ξ_1_old)

    return (λR * ξ_0_old) - (Ω/2)*ξ_1_old[N]

end

function f_η_1(j, Ω, ξ_1_old)

    return (-Ω/2) * ξ_1_old[j]

end

function f_ξ_11(λR, η_2_old, Ω, η_1_old)

    return (λR * η_2_old[1,1]) + (Ω/2)*η_1_old[1]

end

function f_ξ_1N(λL, η_2_old, Ω, η_1_old)

    return (λL * η_2_old[1,N]) + (Ω/2)*η_1_old[N]

end

function f_ξ_1(j, λL, η_2_old, λR ,Ω ,η_1_old)

    return (λL*η_2_old[j,1]) + (λR*η_2_old[j,N]) + (Ω/2)*η_1_old[j]

end

function f_η_2j1(j, λL, ξ_1_old, Ω, ξ_2_old)

    return (λL*ξ_1_old[j]) - (Ω/2)*ξ_2_old[j,1]

end

function f_η_2jN(j, λR, ξ_1_old, Ω, ξ_2_old)

    return (λR*ξ_1_old[j]) - (Ω/2)*ξ_2_old[j,N]

end

function f_η_2(j, Ω, ξ_2_old, k)

    return (-Ω/2)*ξ_2_old[j,k]

end

function f_ξ_2(Ω, η_2_old, j, k)

    return (Ω/2)*η_2_old[j,k]

end