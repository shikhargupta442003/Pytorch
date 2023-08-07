import random
print('Welcome to the PyPassword Generator!')
no_letters=int(input("How many letters would you like in your password?"))
no_symbols=int(input('How many symbols would you like?'))
no_numbers=int(input('How many numbers would you like'))
letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
symbols=['!','@','#','$','%','^','&','*','(',')','_','-','+','=']
numbers=['1','2','3','4','5','6','7','8','9','0']
passward=[]
for i in range(no_numbers):
    passward.append(random.choice(letters))
for i in range(no_symbols):
    passward.append(random.choice(symbols))
for i in range(no_numbers):
    passward.append(random.choice(numbers))
print(passward)
random.shuffle(passward)
print(passward)
print(f"Your password is: {''.join(passward)}")
