print('Welcome to the tip calculator')
bill=float(input('What was the total bill? $'))
people=int(input('How many people will the bill be split?'))
tip=int(input('What percentage would you like to give the tip? 10,12 or 15'))
bill_per_person=(((100+tip)/100)*bill/people)
bill_per_person=round(bill_per_person,2)
print(f'Each person should pay: ${bill_per_person}')