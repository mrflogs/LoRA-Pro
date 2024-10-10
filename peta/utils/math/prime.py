def is_prime(number: int):
    """
    Args:
        number (int): positive integer 'number'

    Returns:
        bool: true if 'number' is prime otherwise false.
    """
    import math  # for function sqrt

    # precondition
    assert isinstance(number, int) and (
        number >= 0
    ), "'number' must been an int and positive"

    status = True

    # 0 and 1 are none primes.
    if number <= 1:
        status = False

    for divisor in range(2, int(round(math.sqrt(number))) + 1):
        # if 'number' divisible by 'divisor' then sets 'status'
        # of false and break up the loop.
        if number % divisor == 0:
            status = False
            break

    # precondition
    assert isinstance(status, bool), "'status' must been from type bool"

    return status
