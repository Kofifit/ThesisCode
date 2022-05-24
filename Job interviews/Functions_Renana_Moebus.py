
def isempty(array):

    # Input: an array
    # Output: False or True
    # This function checks if an array is empty
    # I assumed I filled the array with empty strings (Which is empty), only strings with a symbol are considered full

    # Define output as True, assuming the array is empty until proven otherwise.
    output = True

    # If the value is not empty, we can determine the array is not empty
    if len(array) == 1 and not array[0] == '':
        output = False  # Define output as false
        return output
    # If the first value is empty and the array has more than one value, continue the recursion
    elif len(array) > 1:
        newIndex = int(len(array)/2)  # Find the middle index of the array
        output = isempty(array[0:newIndex]) and isempty(array[newIndex:])  # Apply recursion and find if these two arrays are empty
        return output

    return output


def isfull(array):
    # Input: an array
    # Output: False or True
    # This function checks if an array is full
    # I assumed I filled the array with empty strings (Which is empty), only strings with a symbol are considered full
    # For example: emptyList = ['','','','',''] or []

    # Define output as True, assuming the array is full until proven otherwise
    output = True

    # If the value is empty, we can determine the array is not full
    if len(array) == 1 and array[0] == '':
        output = False  # Define output as false
        return output
    # If the first value is not empty and the array has more than one value, continue the recursion
    elif len(array) > 1:
        newIndex = int(len(array)/2)  # Find the middle index of the array
        output = isfull(array[0:newIndex]) and isfull(array[newIndex:])  # Apply recursion and find if these two arrays are full
        return output

    return output

def helper(array, func):

    # In case an empty list is given

    # isempty() function
    if len(array) == 0 and func == 'isempty':
        output = True
        return output
    elif func == 'isempty':
        return isempty(array)

    # isfull() function
    if len(array) == 0 and func == 'isfull':
        output = False
        return output
    elif func == 'isfull':
        return isfull(array)


## Test runs ##

# isempty() function
arrays = [[], [''], [1], [1,1], [1,1,1,1,1,1,1,1,1,1,1], ['','','','','','','','','','','','','','',''], ['','','','','','','',22,'','','','','','','']]
answers = [True, True, False, False, False, True, False]
index = range(0, len(arrays))
result = True
for i in index:
    result = result and answers[i] == helper(arrays[i], 'isempty')
print(result)

# isfull() function
arrays = [[], ['1'], [1,''], [1,1], [1,1,1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,'',1,1,1], list(range(0,250))]
answers = [False, True, False, True, True, False, True]
index = range(0, len(arrays))
result = True
for i in index:
    result = result and answers[i] == helper(arrays[i], 'isfull')
print(result)


