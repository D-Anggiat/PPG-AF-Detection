import tensorflow as tf

interp = tf.lite.Interpreter(model_path="af_cnn_int8_v2.tflite")
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

print("=== INPUT ===")
print("shape:", inp['shape'])
print("dtype:", inp['dtype'])
print("quant (scale, zero_point):", inp['quantization'])
print()
print("=== OUTPUT ===")
print("shape:", out['shape'])
print("dtype:", out['dtype'])
print("quant (scale, zero_point):", out['quantization'])