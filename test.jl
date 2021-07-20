#= 
Performance tester

=# 

function adder1()

    z::Array{ComplexF64} = zeros(1,10^7)
    z_mod::Float64 = 0

    for i in 1:10^7

        z[i] = rand() + im*rand()

    end

    for i in z

        z_mod += sqrt(conj(i) * i)

    end

    print(z_mod)

end

function adder2()

    z::Array{ComplexF64} = zeros(1,10^7)
    z_mod::Float64 = 0

    for i in 1:10^7

        z[i] = rand() + im*rand()

    end

    for i in z

        z_mod += sqrt(real(i)^2 + imag(i)^2)

    end

    print(z_mod)

end

@time total = adder2()