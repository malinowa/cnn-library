using MLDatasets
using Random
include("conv.jl")
include("maxpool.jl")
include("softmax.jl")

train_images, train_labels = MNIST(:train)[1:1000]
test_images, test_labels = MNIST(:test)[1:1000]

input_image_h = 28
input_image_w = 28
number_of_filters = 20
filter_size = 3
pool_size = 3
output_neurons_number = 10
epochs_number = 10

softmax_input_size =  number_of_filters *  div((input_image_h - (filter_size-1)), pool_size) *  div((input_image_w - (filter_size-1)), pool_size)# 8 x 13 x 13 --> 1 x 10
softmax_input_size = Int(softmax_input_size)

conv = Conv(number_of_filters, filter_size)                 # 28x28x1 -> 26x26x8
pool = MaxPool(pool_size)                                   # 26x26x8 -> 13x13x8
softmax = Softmax(softmax_input_size, output_neurons_number) # 13x13x8 -> 10

function forward(image, label)
  out = forward(conv, image .- 0.5)
  out = forward(pool, out)
  out = forward(softmax, out)

  loss = -log(out[label + 1])
  # println(argma)
  acc = argmax(out)[1] - 1 == label ? 1 : 0

  return out, loss, acc
end

function train(im, label, lr=0.005)
  out, loss, acc = forward(im, label)

  gradient = zeros(10)
  gradient[label + 1] = -1 / out[label + 1]

  gradient = backprop(softmax, gradient, lr)
  gradient = backprop(pool, gradient)
  gradient = backprop(conv, gradient, lr)

  return loss, acc
end

println("MNIST CNN initialized!")

for epoch in 1:epochs_number
  println("--- Epoch ", epoch, " ---")

  global train_images
  global train_labels

  indices = randperm(length(train_labels))
  train_images = train_images[:, :, indices]
  train_labels = train_labels[indices]

  loss = 0
  num_correct = 0
  for (i, (im, label)) in enumerate(zip(eachslice(train_images, dims=3), train_labels))
      if i % 100 == 0
          println(
            "Step ", i, " Past 100 steps: Average Loss ", loss / 100, " | Accuracy: ", num_correct, "%"
          )
          loss = 0
          num_correct = 0
      end

      l, acc = train(im, label)
      loss += l
      num_correct += acc
  end
end

println("\n--- Testing the CNN ---")
loss = 0
num_correct = 0

for i in eachindex(test_labels)
  _ , l, acc = forward(test_images[:, :, i], test_labels[i])
  global loss += l
  global num_correct += acc
end

num_tests = length(test_labels)
println("Test Loss: ", loss / num_tests)
println("Test Accuracy: ", num_correct / num_tests)