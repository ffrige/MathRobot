"""
The SpeakOut function receives a list of objects detected by the camera,
each one with a single digit/character and its position in space.

The mathematical operation is first decoded, then evaluated and finally
spoken out loud.
"""

import winsound
import sys

def decode(objects):

    digits = [str(element[0]) for element in objects]
        
    #look for operation
    # '10' is +    '11' is -    '12' is *    '13' is /
    op = [i for i in digits if int(i)>9]
    if len(op) != 1:
        print("No correct mathematical operation found!")
        return None,None,None,None
    else:
        op_idx = digits.index(op[0])

    number1 = ''.join(digits[:op_idx])
    number2 = ''.join(digits[op_idx+1:])

    #calculate result of operation
    if op[0]=='10':
        result = int(number1) + int(number2)
        op[0]='+'
    elif op[0]=='11':
        result = int(number1) - int(number2)
        op[0]='-'
    elif op[0]=='12':
        result = int(number1) * int(number2)
        op[0]='x'
    elif op[0]=='13':
        result = int(number1) // int(number2)
        op[0]='%'

    return number1, op[0], number2, '=', str(result)


def speak(objects):

    #decode digits into numbers, operation, and result
    members = decode(objects)

    if None in members:
        print("No correct mathematical operation found!")
        return None

    #speak out loud each member
    for member in members:

        if member.isdigit():
            digits = len(member)-1

            for i in member:
                str1 = str(i)
                str2 = str(10**digits)
                if ((int(i)!=1)or(digits!=1))and((int(i)!=0)or(digits!=0)):
                    winsound.PlaySound('voice/'+str1+'.wav', winsound.SND_FILENAME)
                if (int(i)!=0)and(digits!= 0):
                    winsound.PlaySound('voice/'+str2+'.wav', winsound.SND_FILENAME)
                digits -=1
        else:
            winsound.PlaySound('voice/'+member+'.wav', winsound.SND_FILENAME)


if __name__ == "__main__":
    objects = [[6,0,0,0,0],[4,0,0,0,0],[12,0,0,0,0],[3,0,0,0,0]] #64*3
    speak(objects)  #64*3=192
