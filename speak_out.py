import winsound
import sys

str0 = str(input('Operazione: '))

endIndex = str0.find("=")

if endIndex < 0:    #no "=" found -> quit program
    sys.exit(0)
    
str0 = str0[:endIndex]
                
operIndex = str0.find("+")
operTool = 0
if operIndex<0:
    operIndex = str0.find("-")
    operTool = 1
if operIndex<0:
    operIndex = str0.find("*")
    operTool = 2
if operIndex<0:
    operIndex = str0.find("/")
    operTool = 3
if operIndex<0:
    sys.exit(0)

members=[]
members.append(str0[:operIndex])
members.append(str0[operIndex])
members.append(str0[operIndex+1:])
members.append("=")

#TODO replace int with float

if operTool==0:
    members.append(str(int(members[0])+int(members[2])))
elif operTool==1:
    members.append(str(int(members[0])-int(members[2])))
elif operTool==2:
    members.append(str(int(members[0])*int(members[2])))
    members[1] = 'x'
elif operTool==3:
    members.append(str(int(members[0])/int(members[2])))
    members[1] = '%'

print (members[4])


for member in members:

    #TODO read floats
    if member.isdigit():
        digits = len(member)-1

        for i in member:
            str1 = str(i)
            str2 = str(10**digits)
            if ((int(i)!=1)or(digits!=1))and((int(i)!=0)or(digits!=0)):
                winsound.PlaySound('fast/'+str1+'.wav', winsound.SND_FILENAME)
            if (int(i)!=0)and(digits!= 0):
                winsound.PlaySound('fast/'+str2+'.wav', winsound.SND_FILENAME)
            digits -=1
    else:
        winsound.PlaySound('fast/'+member+'.wav', winsound.SND_FILENAME)
