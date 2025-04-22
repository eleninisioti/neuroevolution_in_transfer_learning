import jax.numpy as jnp

def normalize_columns(arr):
    # Get the min and max values for each column
    col_min = jnp.min(arr, axis=0)
    col_max = jnp.max(arr, axis=0) + 0.001

    # Scale to [0, 1] range
    arr_0_1 = (arr - col_min) / (col_max - col_min)

    # Scale to [-1, 1] range
    arr_neg1_1 = arr_0_1 * 2 - 1

    return arr_neg1_1


def get_diversity(data):
    # Step 1: Calculate the standard deviation for each column
    std_devs = jnp.std(data, axis=0)

    # Step 2: Calculate the mean for each column
    means = jnp.abs(jnp.mean(data, axis=0))

    # Step 3: Compute the coefficient of variation for each column
    coeffs_of_variation = std_devs / means

    # Step 4: Take the mean of these coefficients of variation across columns
    mean_cv = jnp.mean(coeffs_of_variation)
    return mean_cv

def shift_network_output(connect_matrix, n_input, n_output):
    start_col = 1 + n_input
    end_col = 1 + n_input + n_output
    columns_to_move = connect_matrix[:, start_col:end_col]
    # Extract the remaining columns
    remaining_columns = jnp.concatenate((connect_matrix[:, :start_col],
                                         connect_matrix[:, end_col:]), axis=1)
    # Concatenate the remaining columns with the columns to move
    w = jnp.concatenate((remaining_columns, columns_to_move), axis=1)

    rows_to_move = connect_matrix[start_col:end_col, :]
    # Extract the remaining columns
    remaining_columns = jnp.concatenate((w[:start_col, :], connect_matrix[end_col:, :]), axis=0)
    # Concatenate the remaining columns with the columns to move
    connect_matrix = jnp.concatenate((remaining_columns, rows_to_move), axis=0)
    return connect_matrix