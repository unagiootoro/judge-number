require "js"
require "base64"

include DNN::Models
include DNN::Layers
include DNN::Optimizers
include DNN::Losses
include DNN::Loaders

Window = JS.global
Document = JS.global[:document]

$trained_mnist_params = nil

JS::Object.undef_method(:then)

def create_model
  model = Sequential.new

  model << InputLayer.new([28, 28, 1])
  model << Flatten.new

  model << Dense.new(64)
  model << BatchNormalization.new
  model << ReLU.new

  model << Dense.new(64)
  model << BatchNormalization.new
  model << ReLU.new

  model << Dense.new(64)
  model << BatchNormalization.new
  model << ReLU.new

  model << Dense.new(10)

  model.setup(Adam.new, SoftmaxCrossEntropy.new)

  model.add_lambda_callback(:after_train) do
    accuracy, loss = model.evaluate($x_test, $y_test)
    BrowserConsole.puts "accuracy: #{accuracy}"
    BrowserConsole.puts "loss: #{loss}"
  end

  model
end

def start_training(model, x_train, y_train, x_test, y_test)
  trainer = ModelTrainer.new(model)
  trainer.start_train(x_train, y_train, 3, batch_size: 128, test: [x_test, y_test], io: BrowserConsole)
  func = -> do
    trainer.update
    if trainer.training?
      JS.global.call(:setTimeout, JS.try_convert(func))
    else
      BrowserConsole.puts("End MLP model training")
    end
  end
  JS.global.call(:setTimeout, JS.try_convert(func))
end

def load_conv2d_model
  model = ConvNet.create([28, 28, 1])
  model.predict1(Numo::SFloat.zeros(28, 28, 1))
  loader = MarshalLoader.new(model)
  loader.load_bin($trained_mnist_params)
  model
end

def update_result(classification)
  str = ""
  10.times do |i|
    str += "#{i}: #{(classification[i] * 100).round(2)}%<br>"
  end
  $result_area[:innerHTML] = str
end

def main
  $model = create_model

  Document.write(<<-EOS)
<canvas id="draw" width=256 height=256></canvas>
<button id="judge">Judge</button>
<button id="clear">Clear</button>
<div id="trainOrLoad">
  <button id="startTraining">Start training for MLP</button><br><br>
  <button id="loadModel">Load conv2d model</button>
</div>
<p id="logField"></p>
<p id="result"></p>
  EOS

  $draw_canvas = Document.getElementById("draw")
  $draw_context = $draw_canvas.getContext("2d")
  $draw_context.fillRect(0, 0, $draw_canvas[:width], $draw_canvas[:height])

  $judge_button = Document.getElementById("judge")
  $clear_button = Document.getElementById("clear")
  $start_training_button = Document.getElementById("startTraining")
  $load_model_button = Document.getElementById("loadModel")

  $result_area = Document.getElementById("result")

  $log_field = Document.getElementById("logField")
  BrowserConsole.dom_element = $log_field

  $judge_button.addEventListener("click") do
    canvas = Document.createElement("canvas");
    canvas[:width] = 28
    canvas[:height] = 28
    ctx = canvas.getContext("2d")
    ctx.drawImage($draw_canvas, 0, 0, canvas[:width], canvas[:height])
    data = ctx.getImageData(0, 0, canvas[:width], canvas[:height])[:data]
    x = Numo::UInt8.cast(data.to_s.split(",").map { |s| s.to_i }).reshape(28, 28, 4)
    x = Numo::SFloat.cast(x[true, true, 0..2]) / 255.0
    x = x.mean(axis: 2, keepdims: true)
    y = $model.predict1(x)
    update_result(y)
  end

  $clear_button.addEventListener("click") do
    $draw_context[:fillStyle] = "black"
    $draw_context.fillRect(0, 0, $draw_canvas[:width], $draw_canvas[:height])
    $result_area[:innerHTML] = ""
  end

  $start_training_button.addEventListener("click") do
    Document[:body].removeChild(Document.getElementById("trainOrLoad"))
    start_training($model, $x_train, $y_train, $x_test, $y_test)
  end

  $load_model_button.addEventListener("click") do
    Document[:body].removeChild(Document.getElementById("trainOrLoad"))
    $model = load_conv2d_model
    BrowserConsole.puts("Load conv model")
  end

  $mouse_down = false

  Window.addEventListener("mousedown") do |e|
    $mouse_down = true
  end

  Window.addEventListener("mouseup") do |e|
    $mouse_down = false
  end

  $draw_canvas.addEventListener("mousemove") do |e|
    if $mouse_down
      rect = e[:target].getBoundingClientRect
      x = e[:clientX].to_s.to_i - 10 - rect[:left].to_s.to_i
      y = e[:clientY].to_s.to_i - 10 - rect[:top].to_s.to_i
      $draw_context[:fillStyle] = "white"
      $draw_context.fillRect(x, y, 20, 20)
    end
  end
end

def boot
  JS.global.fetch("mnist_data.marshal.txt").then do |response|
    response.text.then do |text|
      (x_train, y_train, x_test, y_test) = Marshal.load(Base64.decode64(text.to_s))

      x_train = Numo::SFloat.cast(x_train) / 255
      x_test = Numo::SFloat.cast(x_test) / 255

      y_train = DNN::Utils.to_categorical(y_train, 10, Numo::SFloat)
      y_test = DNN::Utils.to_categorical(y_test, 10, Numo::SFloat)

      $x_train = x_train
      $y_train = y_train
      $x_test = x_test
      $y_test = y_test

      JS.global.fetch("trained_mnist_params.marshal.txt").then do |response|
        response.text.then do |text|
          $trained_mnist_params = Base64.decode64(text.to_s)
          Document[:body].removeChild(Document.getElementById("nowLoading"))
          main
        end
      end
    end
  end
end

boot
