#=

Little Boxes Multi

Contains the little box simulation with the feed back loop. 
This has the RK method used. 

Author: Alex Chapple

Edits:
13/7/2021 - Initial Writing
19/7/2021 - Adding RK (new file)

=#

# Importing Libraries 
using LinearAlgebra 
using Plots
using Random
include("differential_equations.jl")


### Methods --------------------------------------------------------------------------------

function modulo(z)
    return sqrt(real(z)^2 + imag(z)^2)
end

function return_total(η_0_new, ξ_0_new, η_1_new, ξ_1_new, η_2_new, ξ_2_new)

    total::Float64 = modulo(η_0_new) + modulo(ξ_0_new)

    for i in 1:N
        total += (modulo(η_1_new[i]) + modulo(ξ_1_new[i]))
    end

    for j in 1:N
        for k in 1:N
            total += (modulo(η_2_new[j,k]) + modulo(ξ_2_new[j,k]))
        end
    end

    return total 

end

function RK4(Ω, λL, λR, dt, η_0_old, ξ_0_old, η_1_old, ξ_1_old, η_2_old, ξ_2_old)

    # Initialises k1 values 
    η_0_k1::ComplexF64 = 0
    ξ_0_k1::ComplexF64 = 0 
    η_1_k1::Array{ComplexF64} = zeros(1,N)
    ξ_1_k1::Array{ComplexF64} = zeros(1,N)
    η_2_k1::Array{ComplexF64} = zeros(N,N)
    ξ_2_k1::Array{ComplexF64} = zeros(N,N)

    # Calculates k1 values using the respective functions for each coefficient 
    η_0_k1 = f_η_0(Ω, ξ_0_old)
    ξ_0_k1 = f_ξ_0(λL, η_1_old, λR, η_0_old)

    η_1_k1[1] = f_η_11(λL, ξ_0_old, Ω, ξ_1_old)
    η_1_k1[N] = f_η_1N(λR, ξ_0_old, Ω, ξ_1_old)

    ξ_1_k1[1] = f_ξ_11(λR, η_2_old, Ω, η_1_old)
    ξ_1_k1[N] = f_ξ_1N(λL, η_2_old, Ω, η_1_old)

    for j in 2:N-1
        η_1_k1[j] = f_η_1(j, Ω, ξ_1_old)
        ξ_1_k1[j] = f_ξ_1(j, λL, η_2_old, λR, Ω, η_1_old)
    end

    for j in 2:N
        η_2_k1[j,1] = f_η_2j1(j, λL, ξ_1_old, Ω, ξ_2_old)
    end

    for j in 1:N-1
        η_2_k1[j,N] = f_η_2jN(j, λR, ξ_1_old, Ω, ξ_2_old)
    end

    for j in 1:N-2
        for k in (j+1):N-1
            η_2_k1[j,k] = f_η_2(j, Ω, ξ_2_old, k)
        end
    end

    for j in 1:N-1
        for k in (j+1):N
            ξ_2_k1[j,k] = f_ξ_2(Ω, η_2_old, j, k)
        end
    end


    # Initialises k2 values 
    η_0_k2 = 0
    ξ_0_k2 = 0 
    η_1_k2::Array{ComplexF64} = zeros(1,N)
    ξ_1_k2::Array{ComplexF64} = zeros(1,N)
    η_2_k2::Array{ComplexF64} = zeros(N,N)
    ξ_2_k2::Array{ComplexF64} = zeros(N,N)

    # Calculates k2 values using the respective functions for each coefficient 
    η_0_k2 = f_η_0(Ω, ξ_0_old + (dt*ξ_0_k1/2))
    ξ_0_k2 = f_ξ_0(λL, η_1_old + (dt*η_1_k1/2), λR, η_0_old + (dt*η_0_k1/2))

    η_1_k2[1] = f_η_11(λL, ξ_0_old + (dt*ξ_0_k1/2), Ω, ξ_1_old + (dt*ξ_1_k1/2))
    η_1_k2[N] = f_η_1N(λR, ξ_0_old + (dt*ξ_0_k1/2), Ω, ξ_1_old + (dt*ξ_1_k1/2))

    ξ_1_k2[1] = f_ξ_11(λR, η_2_old + (dt*η_2_k1/2), Ω, η_1_old + (dt*η_1_k1/2))
    ξ_1_k2[N] = f_ξ_1N(λL, η_2_old + (dt*η_2_k1/2), Ω, η_1_old + (dt*η_1_k1/2))

    for j in 2:N-1
        η_1_k2[j] = f_η_1(j, Ω, ξ_1_old + (dt*ξ_1_k1/2))
        ξ_1_k2[j] = f_ξ_1(j, λL, η_2_old + (dt*η_2_k1/2), λR, Ω, η_1_old + (dt*η_1_k1/2))
    end

    for j in 2:N
        η_2_k2[j,1] = f_η_2j1(j, λL, ξ_1_old + (dt*ξ_1_k1/2), Ω, ξ_2_old + (dt*ξ_2_k1/2))
    end

    for j in 1:N-1
        η_2_k2[j,N] = f_η_2jN(j, λR, ξ_1_old + (dt*ξ_1_k1/2), Ω, ξ_2_old + (dt*ξ_2_k1/2))
    end

    for j in 1:N-2
        for k in (j+1):N-1
            η_2_k2[j,k] = f_η_2(j, Ω, ξ_2_old + (dt*ξ_2_k1/2), k)
        end
    end

    for j in 1:N-1
        for k in (j+1):N
            ξ_2_k2[j,k] = f_ξ_2(Ω, η_2_old + (dt*ξ_2_k1/2), j, k)
        end
    end

    # Initialises k3 values 
    η_0_k3 = 0
    ξ_0_k3 = 0 
    η_1_k3::Array{ComplexF64} = zeros(1,N)
    ξ_1_k3::Array{ComplexF64} = zeros(1,N)
    η_2_k3::Array{ComplexF64} = zeros(N,N)
    ξ_2_k3::Array{ComplexF64} = zeros(N,N)

    # Calculates k2 values using the respective functions for each coefficient 
    η_0_k3 = f_η_0(Ω, ξ_0_old + (dt*ξ_0_k2/2))
    ξ_0_k3 = f_ξ_0(λL, η_1_old + (dt*η_1_k2/2), λR, η_0_old + (dt*η_0_k2/2))

    η_1_k3[1] = f_η_11(λL, ξ_0_old + (dt*ξ_0_k2/2), Ω, ξ_1_old + (dt*ξ_1_k2/2))
    η_1_k3[N] = f_η_1N(λR, ξ_0_old + (dt*ξ_0_k2/2), Ω, ξ_1_old + (dt*ξ_1_k2/2))

    ξ_1_k3[1] = f_ξ_11(λR, η_2_old + (dt*η_2_k2/2), Ω, η_1_old + (dt*η_1_k2/2))
    ξ_1_k3[N] = f_ξ_1N(λL, η_2_old + (dt*η_2_k2/2), Ω, η_1_old + (dt*η_1_k2/2))

    for j in 2:N-1
        η_1_k3[j] = f_η_1(j, Ω, ξ_1_old + (dt*ξ_1_k2/2))
        ξ_1_k3[j] = f_ξ_1(j, λL, η_2_old + (dt*η_2_k2/2), λR, Ω, η_1_old + (dt*η_1_k2/2))
    end

    for j in 2:N
        η_2_k3[j,1] = f_η_2j1(j, λL, ξ_1_old + (dt*ξ_1_k2/2), Ω, ξ_2_old + (dt*ξ_2_k2/2))
    end

    for j in 1:N-1
        η_2_k3[j,N] = f_η_2jN(j, λR, ξ_1_old + (dt*ξ_1_k2/2), Ω, ξ_2_old + (dt*ξ_2_k2/2))
    end

    for j in 1:N-2
        for k in (j+1):N-1
            η_2_k3[j,k] = f_η_2(j, Ω, ξ_2_old + (dt*ξ_2_k2/2), k)
        end
    end

    for j in 1:N-1
        for k in (j+1):N
            ξ_2_k3[j,k] = f_ξ_2(Ω, η_2_old + (dt*ξ_2_k2/2), j, k)
        end
    end


    # Initialises k4 values 
    η_0_k4 = 0
    ξ_0_k4 = 0 
    η_1_k4::Array{ComplexF64} = zeros(1,N)
    ξ_1_k4::Array{ComplexF64} = zeros(1,N)
    η_2_k4::Array{ComplexF64} = zeros(N,N)
    ξ_2_k4::Array{ComplexF64} = zeros(N,N)

    # Calculates k2 values using the respective functions for each coefficient 
    η_0_k4 = f_η_0(Ω, ξ_0_old + (dt*ξ_0_k3))
    ξ_0_k4 = f_ξ_0(λL, η_1_old + (dt*η_1_k3), λR, η_0_old + (dt*η_0_k3))

    η_1_k4[1] = f_η_11(λL, ξ_0_old + (dt*ξ_0_k3), Ω, ξ_1_old + (dt*ξ_1_k3))
    η_1_k4[N] = f_η_1N(λR, ξ_0_old + (dt*ξ_0_k3), Ω, ξ_1_old + (dt*ξ_1_k3))

    ξ_1_k4[1] = f_ξ_11(λR, η_2_old + (dt*η_2_k3), Ω, η_1_old + (dt*η_1_k3))
    ξ_1_k4[N] = f_ξ_1N(λL, η_2_old + (dt*η_2_k3), Ω, η_1_old + (dt*η_1_k3))

    for j in 2:N-1
        η_1_k4[j] = f_η_1(j, Ω, ξ_1_old + (dt*ξ_1_k3))
        ξ_1_k4[j] = f_ξ_1(j, λL, η_2_old + (dt*η_2_k3), λR, Ω, η_1_old + (dt*η_1_k3))
    end

    for j in 2:N
        η_2_k4[j,1] = f_η_2j1(j, λL, ξ_1_old + (dt*ξ_1_k3), Ω, ξ_2_old + (dt*ξ_2_k3))
    end

    for j in 1:N-1
        η_2_k4[j,N] = f_η_2jN(j, λR, ξ_1_old + (dt*ξ_1_k3), Ω, ξ_2_old + (dt*ξ_2_k3))
    end

    for j in 1:N-2
        for k in (j+1):N-1
            η_2_k4[j,k] = f_η_2(j, Ω, ξ_2_old + (dt*ξ_2_k3), k)
        end
    end

    for j in 1:N-1
        for k in (j+1):N
            ξ_2_k4[j,k] = f_ξ_2(Ω, η_2_old + (dt*ξ_2_k3), j, k)
        end
    end


    # Summarises all the results 
    η_0_new::ComplexF64 = η_0_old + (dt/6)*(η_0_k1 + 2*η_0_k2 + 2*η_0_k3 + η_0_k4)
    ξ_0_new::ComplexF64 = ξ_0_old + (dt/6)*(ξ_0_k1 + 2*ξ_0_k2 + 2*ξ_0_k3 + ξ_0_k4)

    η_1_new::Array{ComplexF64} = η_1_old + (dt/6)*(η_1_k1 + 2*η_1_k2 + 2*η_1_k3 + η_1_k4)
    ξ_1_new::Array{ComplexF64} = ξ_1_old + (dt/6)*(ξ_1_k1 + 2*ξ_1_k2 + 2*ξ_1_k3 + ξ_1_k4)

    η_2_new::Array{ComplexF64} = η_2_old + (dt/6)*(η_2_k1 + 2*η_2_k2 + 2*η_2_k3 + η_2_k4)
    ξ_2_new::Array{ComplexF64} = ξ_2_old + (dt/6)*(ξ_2_k1 + 2*ξ_2_k2 + 2*ξ_2_k3 + ξ_2_k4)

    return η_0_new, ξ_0_new, η_1_new, ξ_1_new, η_2_new, ξ_2_new

end

function evolve(time_list, Ω, γL, γR , dt, τ, phase, N)

    #=
    Evolves one simulation from start to finish 

    =#

    # Initial conditions
    η_0_list::Array{ComplexF64} = [1]
    ξ_0_list::Array{ComplexF64} = [0]
    η_1_list::Array{Array{ComplexF64}} = [zeros(1,N)]
    ξ_1_list::Array{Array{ComplexF64}} = [zeros(1,N)]
    η_2_list::Array{Array{ComplexF64}} = [zeros(N,N)]
    ξ_2_list::Array{Array{ComplexF64}} = [zeros(N,N)]

    λL = exp(1im * phase / 2) * sqrt(γL) * sqrt(N/τ)
    λR = exp(-1im * phase / 2) * sqrt(γR) * sqrt(N/τ)

    for index in 1:size(time_list)[1]

        # η_0_old = copy(η_0_list[end]) NO NEED FOR COPYING I THINK 
        # ξ_0_old = copy(ξ_0_list[end])
    
        # η_1_old = copy(η_1_list[end])
        # ξ_1_old = copy(ξ_1_list[end])
    
        # η_2_old = copy(η_2_list[end])
        # ξ_2_old = copy(ξ_2_list[end])

        η_0_old::ComplexF64 = η_0_list[end]
        ξ_0_old::ComplexF64 = ξ_0_list[end]
    
        η_1_old::Array{ComplexF64} = η_1_list[end]
        ξ_1_old::Array{ComplexF64} = ξ_1_list[end]
    
        η_2_old::Array{ComplexF64} = η_2_list[end]
        ξ_2_old::Array{ComplexF64} = ξ_2_list[end]
    
        η_0_new, ξ_0_new, η_1_new, ξ_1_new, η_2_new, ξ_2_new = RK4(Ω, λL, λR, dt, η_0_old, ξ_0_old, η_1_old, ξ_1_old, η_2_old, ξ_2_old)

        # Normalise everything here
        total = return_total(η_0_new, ξ_0_new, η_1_new, ξ_1_new, η_2_new, ξ_2_new)

        η_0_new::ComplexF64 /= total
        ξ_0_new::ComplexF64 /= total 
        η_1_new::Array{ComplexF64} /= total
        ξ_1_new::Array{ComplexF64} /= total 
        η_2_new::Array{ComplexF64} /= total
        ξ_2_new::Array{ComplexF64} /= total

        if index % 10 == 0 # This is in accordance to the long time step. Only checks to see if there is a photon every 10 dt steps

            # Do statistics here --------------------------------------------------------------------------

            # Here ψ_0 = <ψ_0 | ψ_0>
            ψ_0::Float64 = modulo(η_0_new)^2 + modulo(ξ_0_new)^2
            
            for j in 1:N-1
                ψ_0 += (modulo(η_1_new[j])^2 + modulo(ξ_1_new[j]))^2
            end

            for j in 1:N-2
                for k in (j+1):N-1
                    ψ_0 += (modulo(η_2_new[j,k])^2 + modulo(ξ_2_new[j,k]))^2
                end
            end

            # Here ψ_1 = <ψ_1 | ψ_1>
            ψ_1::Float64 = modulo(η_1_new[N])^2 + modulo(ξ_1_new[N])^2

            for j in 1:N-1
                ψ_1 += (modulo(η_2_new[j,N])^2 + modulo(ξ_2_new[j,N]))^2
            end

            # Probablity for observing a click
            prob = ψ_1 / (ψ_1 + ψ_0)
            rand_num = rand()

            if rand_num <= prob # Photon found

                η_0::ComplexF64 = η_1_new[N]
                ξ_0::ComplexF64 = ξ_1_new[N]

                η_1::Array{ComplexF64} = zeros(1,N)
                ξ_1::Array{ComplexF64} = zeros(1,N)

                η_1[1] = 0
                ξ_1[1] = 0

                for j in 2:N
                    η_1[j] = η_2_new[j-1,N]
                    ξ_1[j] = ξ_2_new[j-1,N]
                end

                η_2::Array{ComplexF64} = zeros(N,N) # η2_j,k
                ξ_2::Array{ComplexF64} = zeros(N,N) # ξ2_j,k

                total = return_total(η_0, ξ_0, η_1, ξ_1, η_2, ξ_2)
                η_0 /= total 
                ξ_0 /= total 
                η_1 /= total 
                ξ_1 /= total 
                η_2 /= total
                ξ_2 /= total 

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

                total = return_total(η_0, ξ_0, η_1, ξ_1, η_2, ξ_2)
                η_0 /= total 
                ξ_0 /= total 
                η_1 /= total 
                ξ_1 /= total 
                η_2 /= total
                ξ_2 /= total 


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

            total = return_total(η_0, ξ_0, η_1, ξ_1, η_2, ξ_2) # This may be redundant as it has already been normalised
            η_0 /= total 
            ξ_0 /= total 
            η_1 /= total 
            ξ_1 /= total 
            η_2 /= total
            ξ_2 /= total 

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

function average_simulation(N, phase, Ω, γL, γR, end_time, time_steps)

    time_list = LinRange(0,end_time,time_steps)
    dt = end_time/time_steps
    Δt = 10 * dt
    τ = Δt * N

    avg_spin_up::Array{Float64} = zeros(size(time_list)[1])
    avg_spin_down::Array{Float64} = zeros(size(time_list)[1])

    for sim in 1:num_of_simulations

        η_0_list, ξ_0_list, η_1_list, ξ_1_list, η_2_list, ξ_2_list = evolve(time_list, Ω, γL, γR, dt, τ, phase, N)

        spin_up_prob::Array{Float64} = zeros(size(time_list)[1])
        spin_down_prob::Array{Float64} = zeros(size(time_list)[1])

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

        # if sim % 10 == 0
        #     print(string(sim) * "/" * string(num_of_simulations) * " Simulations Completed.\r")
        # end
        print(string(sim) * "/" * string(num_of_simulations) * " Simulations Completed.\r")

    end

    # Clean up data 
    avg_spin_up /= num_of_simulations
    avg_spin_down /= num_of_simulations

    return time_list, avg_spin_down, avg_spin_up

end

function plot_results(time_list, avg_spin_down, avg_spin_up, Ω, γL, γR, phase, N, end_time, time_steps, num_of_simulations)

    attributes = "spin down," * " Ω:" * string(round(Ω, digits=1)) * " ,γL:" * string(γL) * " ,γR:" *　string(γR) * " ,phase:" * string(phase) * " ,N:" * string(N) * "\n dt = " * string(end_time/time_steps) * " ,sim_num:" * string(num_of_simulations)

    plot(time_list, avg_spin_down, lw=2,label="spin down", dpi=600)
    xlabel!("time")
    ylabel!("prob spin down")
    title!(attributes, titlefont=10)
    name = "Figures/spin_down_RK4_3.png"
    savefig(name)

    attributes = "spin up," * " Ω:" * string(round(Ω, digits=1)) * " ,γL:" * string(γL) * " ,γR:" *　string(γR) * " ,phase:" * string(phase) * " ,N:" * string(N) * "\n dt = " * string(end_time/time_steps) * " ,sim_num:" * string(num_of_simulations)

    plot(time_list, avg_spin_up, lw=2, label="spin up", dpi=600)
    xlabel!("time")
    ylabel!("prob spin up")
    title!(attributes, titlefont=10)
    name = "Figures/spin_up_RK4_3.png"
    savefig(name)

end

time_steps = 10000
end_time = 8
num_of_simulations = 1

Ω = 10π
γL = 0.5
γR = 0.5
phase = π
N = 20

@time time_list, avg_spin_down, avg_spin_up = average_simulation(N, phase, Ω, γL, γR, end_time, time_steps)

plot_results(time_list, avg_spin_down, avg_spin_up, Ω, γL, γR, phase, N, end_time, time_steps, num_of_simulations)