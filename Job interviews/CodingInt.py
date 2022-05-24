# nextIndex = 0
# numArray = list(0, 3)
# #isempty
# #isfull
#
# def insertNumber(value):
#
#
#     if isempty(numArray):
#         firstIndex = 0
#         numArray[0] = value
#         nextIndex = 1
#
#     elif isfull(numArray):
#         numArray[nextIndex] = value
#         firstIndex = nextIndex + 1
#         nextIndex += 1
#
#     else:
#         numArray[nextIndex] = value
#         nextIndex += 1
#
#     output = nextIndex-1
#
#     if nextIndex == len(numArray):
#         nextIndex = 0
#
#     return output
#
#
# def extractNumber():
#     output = numArray[firstIndex]
#     firstIndex += 1
#
#     return output

def isempty(array):

    # Input: an array
    # Output: False or True
    # This function checks if an array is empty
    # I assumed I filled the array with empty strings (Which is empty), only strings with a symbol are considered full

    output = True

    # In case an empty list is given
    if len(array) == 0:
        return output

    Value = array[0]

    if not Value == '':  # If the value of the current index is not empty
        output = False
        # The array is not empty, return answer
        return output
    else:  # If the value of the current index is empty
        if len(array) == 2 and array[1] != '':
            output = False
            return output
        elif len(array) == 1 or len(array) == 2 and array[1] == '':
            return output
        else:
            newIndex = int(len(array)/2)+1
            array1 = array[1:newIndex]
            array2 = array[newIndex:]
            output = isempty(array1) and isempty(array2)
            return output



def isfull(array):
    # Input: an array
    # Output: False or True
    # This function checks if an array is full
    # I assumed I filled the array with empty strings (Which is empty), only strings with a symbol are considered full
    # For example: emptyList = ['','','','',''] or []

    output = True

    # In case an empty list is given
    if len(array) == 0:
        output = False
        return output


    Value = array[0]

    if Value == '':  # If the value of the current index is empty
        output = False
        # The array is not full, return answer
        return output
    else:  # If the value of the current index is not empty
        if len(array) == 2 and array[1] == '':
            output = False
            return output
        elif len(array) == 1 or len(array) == 2 and array[1] != '':
            return output
        else:
            newIndex = int(len(array)/2)+1
            array1 = array[1:newIndex]
            array2 = array[newIndex:]
            output = isfull(array1) and isfull(array2)
            return output




