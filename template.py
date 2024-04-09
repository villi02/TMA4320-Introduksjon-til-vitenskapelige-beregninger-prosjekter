## File containing template for how we want the code to be documented and structured

## How functions should be documented


def function_name(arg1, arg2) -> type:
    """
    Description of the function

    Parameters:
    arg1 (type): description of arg1
    arg2 (type): description of arg2

    Returns:
    type: description of the return value
    """
    # code here
    pass


## How git commits should be structured

# [what] [where] [why] (optional)

# what: what was done
# where: where was it done, what function or task
# why: why was it done, what was the purpose, should only be used if the original method worked but was changed for a reason

# Example
# Added new function to calculate the sum of two numbers in 2 a)

# Example with why
# Changed the function to use vectorization instead of for loop in 2 a), the original method was too slow


## How to structure a branch

# [name of owner]-[task number or method being worked on]

# Example

# William-2a

## How to structure a pull request

# [task number or method being worked on] [what was done]

# Example

# 2a Added new function to calculate the sum of two numbers
