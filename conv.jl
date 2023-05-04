mutable struct Conv
  num_filters::Int
  filter_size::Int
  filters::Array{Float64, 3}
  last_input::Array{Float64, 2}

  function Conv(num_filters::Int, filter_size::Int)
    filters = randn(filter_size, filter_size, num_filters) ./ 9
    new(num_filters, filter_size, filters)
  end
end

function iterate_regions(conv::Conv, image::Array{Float64, 2})
  h, w = size(image)
  channel_size = (h-(conv.filter_size-1)) * (w-(conv.filter_size-1))
  c = Channel{
      Tuple{
          SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}, 
          Int, 
          Int
      }}(channel_size)

  @async begin
      for i in 1:h-(conv.filter_size-1), j in 1:w-(conv.filter_size-1)
          im_region = view(image, i:i+(conv.filter_size-1), j:j+(conv.filter_size-1))
          put!(c, (im_region, i, j))
      end
      close(c)
  end
  
  return c
end

function forward(conv::Conv, input::Array{Float64, 2})
    conv.last_input = input
    h, w = size(input)
    output = zeros(Float64, h-(conv.filter_size-1), w-(conv.filter_size-1), conv.num_filters)

    for (im_region, i, j) in iterate_regions(conv, input)
        output[i, j, :] = sum(im_region .* conv.filters, dims=(1,2))[:]
    end

    return output
end

function backprop(conv::Conv, d_L_d_out, learn_rate)
  d_L_d_filters = zeros(size(conv.filters))
  d_L_d_input = zeros(size(conv.last_input))

  for (im_region, i, j) in iterate_regions(conv, conv.last_input)
    for f in 1:conv.num_filters
      d_L_d_filters[:, :, f] .+= d_L_d_out[i, j, f] .* im_region
      d_L_d_input[i:i+(conv.filter_size-1), j:j+(conv.filter_size-1)] .+= d_L_d_out[i, j, f] .* conv.filters[:, :, f]
    end
  end

  # Update filters
  conv.filters .-= learn_rate .* d_L_d_filters

  return d_L_d_input
end