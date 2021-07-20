using Base: func_for_method_checked
#= 
Performance tester

=# 

function modulo(z)
    return sqrt(real(z)^2 + imag(z)^2)
end


function make_list(N)

    a::Array{ComplexF64} = zeros(N,N)
    b::Array{ComplexF64} = zeros(N,N)

    for i in 1:N
        for j in 1:N
            a[i,j] = i + j*im
            b[i,j] = i*im + j
        end
    end

    return N,a,b

end

function for_loop1(N, a, b)

    total = 0

    for i in 1:N
        for j in 1:N
            total += a[i,j]

            if j !== 1
                total += b[i,j]
            end
        end
    end

    print(total)

end

function for_loop2(N, a, b)

    total = 0

    for i in 1:N
        for j in 1:N
            total += a[i,j]
        end
    end

    for i in 1:N
        for j in 2:N
            total += b[i,j]
        end
    end

    print(total)


end


N,a,b = make_list(10000)

@time for_loop1(N,a,b)