#= 
Performance tester

=# 

function adder1()

    total::Float64 = 10000
    total_list::Array{Float64} = zeros(1,800000)

    for i in 1:800000
        total /= i
        total_list[i] = total 
    end

    return total, total_list

end

function adder2()

    total::Float64 = 10000
    total_list::Array{Float64}  = zeros(1,800000)

    for i in 1:800000
        total = total / i
        total_list[i] = total 
    end

    return total, total_list

end

@time total = adder2()