import sys

sys.path.append(str(Path.cwd() / "lib_nets"))
print(sys.path)


from Nets.EQ_Net_A import NeuralNetwork

model = NeuralNetwork.default_config()
saved_model_file = "models\torch\NH3_net_loguniform.pt"

model.from_state_dict(saved_model_file)

p = T= x_initial = 1

x_eq = model(p, T, x_initial)