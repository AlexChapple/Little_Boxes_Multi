using Base: Float64
#=

Little Boxes Multi

Contains the little box simulation with the feed back loop. 

Author: Alex Chapple

Edits:
13/7/2021 - Initial Writing


=#

# Importing Libraries 
using LinearAlgebra 
using Plots
using Random


### Methods --------------------------------------------------------------------------------

function modulo(z)
    return sqrt(real(z)^2 + imag(z)^2)
end

function evolve(time_list, Γ, γL, γR ,h, phase, N)

    #=
    Evolves one simulation from start to finish 

    =#

    # Initial conditions
    η_0_list::Array{Float64} = [1]
    ξ_0_list::Array{Float64} = [0]
    η_1_list = [zeros(1,N)]
    ξ_1_list = [zeros(1,N)]
    η_2_list = [zeros(N,N)]
    ξ_2_list = [zeros(N,N)]

    λL = exp(1im * phase / 2) * sqrt(γL) * sqrt(N/h)
    λR = exp(-1im * phase / 2) * sqrt(γR) * sqrt(N/h)

    for t in time_list

        η_0_old = copy(η_0_list[end])
        ξ_0_old = copy(ξ_0_list[end])
    
        η_1_old = copy(η_1_list[end])
        ξ_1_old = copy(ξ_1_list[end])
    
        η_2_old = copy(η_2_list[end])
        ξ_2_old = copy(ξ_2_list[end])
    
        η_0_new = 0
        ξ_0_new = 0
        η_1_new = zeros(1,N)
        ξ_1_new = zeros(1,N)
        η_2_new = zeros(N,N) # η2_j,k
        ξ_2_new = zeros(N,N) # ξ2_j,k

        # η0 and ξ0 coeffs update 
        η_0_new = η_0_old +  ((-Γ/2) * ξ_0_old)*h
        ξ_0_new = ξ_0_old + ((λL * η_1_old[1] + λR * η_1_old[N] + (Γ/2)*η_0_old)) * h 

        # η1, ξ1 coeffs update
        η_1_new[1] = η_1_old[1] + (λL * ξ_0_old - (Γ/2)*ξ_1_old[1])*h
        η_1_new[N] = η_1_old[N] + (λR * ξ_0_old - (Γ/2)*ξ_1_old[N])*h

        for j in 2:N-1
            η_1_new[j] = η_1_old[j] + ((-Γ/2) * ξ_1_old[j])*h
        end
 
        ξ_1_new[1] = ξ_1_old[1] + (λR * η_2_old[1,1] + (Γ/2)*η_1_old[1])*h
        ξ_1_new[N] = ξ_1_old[N] + (λL * η_2_old[1,N] + (Γ/2)*η_1_old[N])*h

        for j in 2:N-1
            ξ_1_new[j] = ξ_1_old[j] + (λL*η_2_old[j,1] + λR*η_2_old[j,N] + (Γ/2)*η_1_old[j])*h
        end

        # η2, ξ2 coeffs update
        for j in 2:N
            η_2_new[j,1] = η_2_old[j,1] + (λL*ξ_1_old[j] - (Γ/2)*ξ_2_old[j,1])*h
        end

        for j in 1:N-1
            η_2_new[j,N] = η_2_old[j,N] + (λR*ξ_1_old[j] - (Γ/2)*ξ_2_old[j,N])*h
        end

        for j in 1:N-2
            for k in (j+1):N-1
                η_2_new[j,k] = η_2_old[j,k] + ((-Γ/2)*ξ_2_old[j,k])*h
            end
        end

        for j in 1:N-1
            for k in (j+1):N
                ξ_2_new[j,k] = ξ_2_old[j,k] + ((Γ/2)*η_2_old[j,k])*h
            end
        end

        # Do statistics here --------------------------------------------------------------------------

        # Here ψ_0 = <ψ_0 | ψ_0>
        ψ_0 = modulo(η_0_new)^2 + modulo(ξ_0_new)^2
        
        for j in 1:N-1
            ψ_0 += (modulo(η_1_new[j])^2 + modulo(ξ_1_new[j])^2)
        end

        for j in 1:N-2
            for k in (j+1):N-1
                ψ_0 += (modulo(η_2_new[j,k])^2 + modulo(ξ_2_new[j,k])^2)
            end
        end

        # Here ψ_1 = <ψ_1 | ψ_1>
        ψ_1 = modulo(η_1_new[N])^2 + modulo(ξ_1_new[N])^2

        for j in 1:N-1
            ψ_1 += (modulo(η_2_new[j,N])^2 + modulo(ξ_2_new[j,N])^2)
        end

        # Probablity for observing a click

        if t % 10 == 0

            prob = ψ_1 / (ψ_1 + ψ_0)
            rand_num = rand()

            if rand_num <= prob # Photon found

                η_0 = η_1_new[N]
                ξ_0 = ξ_1_new[N]

                η_1 = zeros(1,N)
                ξ_1 = zeros(1,N)

                η_1[1] = 0
                ξ_1[1] = 0

                for j in 2:N
                    η_1[j] = η_2_new[j-1,N]
                    ξ_1[j] = ξ_2_new[j-1,N]
                end

                η_2 = zeros(N,N) # η2_j,k
                ξ_2 = zeros(N,N) # ξ2_j,k

                # Update list
                push!(η_0_list, η_0)
                push!(ξ_0_list, ξ_0)
                push!(η_1_list, η_1)
                push!(ξ_1_list, ξ_1)
                push!(η_2_list, η_2)
                push!(ξ_2_list, ξ_2)

            else # Photon not found

                η_0 = η_0_new
                ξ_0 = ξ_0_new 

                η_1 = zeros(1,N)
                ξ_1 = zeros(1,N)

                η_1[1] = 0
                ξ_1[1] = 0

                for j in 2:N
                    η_1[j] = η_1_new[j-1]
                    ξ_1[j] = ξ_1_new[j-1]
                end

                η_2 = zeros(N,N) # η2_j,k
                ξ_2 = zeros(N,N) # ξ2_j,k

                for k in 2:N
                    η_2[1,k] = 0
                    ξ_2[1,k] = 0
                end

                for j in 2:N-1
                    for k in (j+1):N
                        η_2[j,k] = η_2_new[j-1,k-1]
                        ξ_2[j,k] = ξ_2_new[j-1,k-1]
                    end
                end

                # Update list
                push!(η_0_list, η_0)
                push!(ξ_0_list, ξ_0)
                push!(η_1_list, η_1)
                push!(ξ_1_list, ξ_1)
                push!(η_2_list, η_2)
                push!(ξ_2_list, ξ_2)


            end

        else

            η_0 = η_0_new
            ξ_0 = ξ_0_new
            η_1 = η_1_new
            ξ_1 = ξ_1_new
            η_2 = η_2_new
            ξ_2 = ξ_2_new 

            # Update list
            push!(η_0_list, η_0)
            push!(ξ_0_list, ξ_0)
            push!(η_1_list, η_1)
            push!(ξ_1_list, ξ_1)
            push!(η_2_list, η_2)
            push!(ξ_2_list, ξ_2)

        end

    end

    return η_0_list, ξ_0_list, η_1_list, ξ_1_list, η_2_list, ξ_2_list

end

function average_simulation(N, phase, Γ, γL, γR, end_time, time_steps)

    time_list = LinRange(0,end_time,time_steps)
    h = end_time/time_steps

    avg_spin_up = zeros(size(time_list)[1])
    avg_spin_down= zeros(size(time_list)[1])

    for sim in 1:num_of_simulations

        η_0_list, ξ_0_list, η_1_list, ξ_1_list, η_2_list, ξ_2_list = evolve(time_list, Γ, γL, γR, h, phase, N)

        spin_up_prob = zeros(size(time_list)[1])
        spin_down_prob = zeros(size(time_list)[1])

        for i in 1:size(time_list)[1]

            # Calculates total spin down probability 
            spin_down_prob[i] += modulo(η_0_list[i])^2

            for j in 1:N
                spin_down_prob[i] += modulo(η_1_list[i][j])^2
            end

            for j in 1:N
                for k in 1:N
                    spin_down_prob[i] += modulo(η_2_list[i][j,k])^2
                end
            end

            # Calculates total spin up probability 
            spin_up_prob[i] += modulo(ξ_0_list[i])^2

            for j in 1:N
                spin_up_prob[i] += modulo(ξ_1_list[i][j])^2
            end

            for j in 1:N
                for k in 1:N
                    spin_up_prob[i] += modulo(ξ_2_list[i][j,k])^2
                end
            end

            total = spin_up_prob[i] + spin_down_prob[i]
            spin_up_prob[i] /= total 
            spin_down_prob[i] /= total 

        end

        avg_spin_down += spin_down_prob
        avg_spin_up += spin_up_prob

        if sim % 10 == 0
            print(string(sim) * "/" * string(num_of_simulations) * " Simulations Completed.\r")
        end

    end

    # Clean up data 
    avg_spin_up /= num_of_simulations
    avg_spin_down /= num_of_simulations

    return time_list, avg_spin_down, avg_spin_up

end

function plot_results(time_list, avg_spin_down, avg_spin_up)

    plot(time_list, avg_spin_down, lw=2,label="spin down", dpi=600)
    plot!(time_list, avg_spin_up, lw=2, label="spin up")
    xlabel!("time")
    ylabel!("prob spin up/down")
    title = "Figures/spin.png"
    savefig(title)

end

time_steps = 500
end_time = 8
num_of_simulations = 1

Γ = 10π
γL = 0.5
γR = 0.5
phase = 0
N = 20

@time time_list, avg_spin_down, avg_spin_up = average_simulation(N, phase, Γ, γL, γR, end_time, time_steps)

plot_results(time_list, avg_spin_down, avg_spin_up)