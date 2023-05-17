
# cost function
def cost_fn(params):
    cost = 0
    for k in range(3):
        cost += torch.abs(circuit(params, Paulis[k]) - bloch_v[k])

    return cost