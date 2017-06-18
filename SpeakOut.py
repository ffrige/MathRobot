import winsound
import sys

"""
10 is +
11 is -
12 is *
13 is /

"""

def speak(result):

    digits = [element[0] for element in result]
    operation = [i for i in digits if i>9]
    if len(operation) != 1:
        print("wrong number of operations found")
        return 0

    return 0 

    #TODO -finish the rest...

    #calculate result of operation
    if operation==10:
        members.append(str(int(members[0])+int(members[2])))
    elif operation==11:
        members.append(str(int(members[0])-int(members[2])))
    elif operation==12:
        members.append(str(int(members[0])*int(members[2])))
    elif operation==13:
        members.append(str(int(members[0])/int(members[2])))


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
