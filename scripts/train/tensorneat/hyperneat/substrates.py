

def n_parity_substrate(n_inputs, n_outputs, n_hidden ):
    input_coords = []
    current_x = -1
    for i in range(n_inputs+1):  # because of bias
        input_coords.append((-1, current_x))
        current_x = current_x + 2 /n_inputs

    output_coords = []
    current_x = -1
    for i in range(n_outputs):
        output_coords.append((1, current_x))
        current_x = current_x + 2 / n_outputs

    hidden_coords = []
    current_x = -1
    for i in range(n_hidden):
        hidden_coords.append((0, current_x))
        current_x = current_x + 2 / n_hidden

    return input_coords, hidden_coords, output_coords


def get_substrate(env_name, n_input, n_output, n_hidden):

    if env_name == "n_parity":
        return n_parity_substrate(n_input, n_output, n_hidden)