
def get_eps():
    stability_epsilon = 1e-07
    return stability_epsilon


def safe_divide(lib, numerator, denominator):
    with lib.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result[denominator == 0] = 0
    return result
