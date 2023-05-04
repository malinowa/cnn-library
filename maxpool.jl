using LinearAlgebra

mutable struct MaxPool
  pool_size::Int
  last_input::Array{Float64, 3}

  function MaxPool(pool_size::Int)
    new(pool_size)
  end
end

function forward(pool::MaxPool, input::Array{Float64, 3})
  pool.last_input = input
  h, w, num_filters = size(input)
  new_h = div(h, pool.pool_size)
  new_w = div(w, pool.pool_size)
  output = zeros(new_h, new_w, num_filters)

  for i in 1:new_h, j in 1:new_w, k in 1:num_filters
    output[i, j, k] = maximum(input[pool.pool_size*i-(pool.pool_size-1):pool.pool_size*i, pool.pool_size*j-(pool.pool_size-1):pool.pool_size*j, k])
  end

  return output
end

function backprop(mp::MaxPool, d_L_d_out)
  d_L_d_input = zeros(size(mp.last_input))

  input = mp.last_input
  h, w, num_filters = size(input)
  new_h = div(h, mp.pool_size)
  new_w = div(w, mp.pool_size)

  for i in 1:new_h, j in 1:new_w, k in 1:num_filters
    im_region = input[mp.pool_size*i-(mp.pool_size-1):mp.pool_size*i, mp.pool_size*j-(mp.pool_size-1):mp.pool_size*j, k]
    h, w = size(im_region)
    max = maximum(im_region)

    for i2 in 1:h
      for j2 in 1:w
        if im_region[i2, j2] == max
          d_L_d_input[i*mp.pool_size - (mp.pool_size-1) + (i2-1), j*mp.pool_size - (mp.pool_size-1) + j2-1, k] = d_L_d_out[i, j, k]
        end
      end
    end

  end

  return d_L_d_input
end