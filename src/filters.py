from weights_utils import scale_model_weights


def acc_filter(weights=None, accuracies=None):
    avg_acc = sum(accuracies)/len(accuracies)
    scaled_local_weight_list_filtered = [
        x for n, x in enumerate(weights) if accuracies[n] > avg_acc
    ]
    # scale the model weights and add to list
    scaling_factor = 1 / len(scaled_local_weight_list_filtered)
    scaled_local_weight_list_filtered = [
        scale_model_weights(x, scaling_factor) for x in scaled_local_weight_list_filtered
    ]
    return scaled_local_weight_list_filtered
