#Uses python3

import sys

def largest_number(a):
    res = ''
    while a:
        max_digit = 0
        max_index = 0
        for idx, num in enumerate(a):
            # Reduce num to the first digit
            num = int(num)
            digit = num
            while digit // 10 != 0:
                digit //= 10
            if digit > max_digit:
                max_digit = digit
                max_index = idx
            elif digit == max_digit:
                digit_1 = num
                digit_2 = int(a[max_index])
                while digit_1 > 99:
                    digit_1 //= 10
                while digit_2 > 99:
                    digit_2 //= 10
                while digit_1 // 10 != 0:
                    digit_1 %= 10
                while digit_2 // 10 != 0:
                    digit_2 %= 10

                if digit_1 > digit_2:
                    max_digit = digit
                    max_index = idx
                elif digit_1 == digit_2:
                    digit_1 = num
                    digit_2 = int(a[max_index])
                    while digit_1 // 10 != 0:
                        digit_1 %= 10
                    while digit_2 // 10 != 0:
                        digit_2 %= 10
                    if digit_1 > digit_2:
                        max_digit = digit
                        max_index = idx



        res += str(a[max_index])
        del a[max_index]
    return res


if __name__ == '__main__':
    input = sys.stdin.read()
    data = input.split()
    a = data[1:]
    print(largest_number(a))
