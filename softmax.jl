using LinearAlgebra

mutable struct Softmax
    weights::Matrix{Float64}
    biases::Vector{Float64}
    last_input_shape::Tuple
    last_input::Vector{Float64}
    last_totals::Matrix{Float64}

    function Softmax(input_len::Int, nodes::Int)
      weights = randn(input_len, nodes) / input_len
      biases = zeros(nodes)
      new(weights, biases)
    end
end

function forward(s::Softmax, input::AbstractArray)
    s.last_input_shape = size(input)
    input = vec(input)

    s.last_input = input

    totals = input' * s.weights .+ s.biases'
    s.last_totals = totals
    exp_vals = exp.(totals)
    return transpose(exp_vals ./ sum(exp_vals))
end

function backprop(s::Softmax, d_L_d_out, learn_rate)
  for (i, gradient) in enumerate(d_L_d_out)
      if gradient == 0
          continue
      end

      t_exp = exp.(s.last_totals)

      S = sum(t_exp)

      # Gradients of out[i] against totals
      d_out_d_t = -t_exp[i] .* t_exp / (S ^ 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ^ 2)

      # Gradients of totals against weights/biases/input
      d_t_d_w = s.last_input
      d_t_d_b = 1
      d_t_d_inputs = s.weights

      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t

      # Gradients of loss against weights/biases/input
      d_L_d_w = d_t_d_w * d_L_d_t
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs * d_L_d_t'

      # Update weights / biases
      s.weights -= learn_rate * d_L_d_w
      s.biases .-= learn_rate * d_L_d_b'
      return reshape(d_L_d_inputs, s.last_input_shape)
  end
end