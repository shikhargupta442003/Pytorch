rock = '''
    _______
---'   ____)
      (_____)
      (_____)
      (____)
---.__(___)
'''

paper = '''
    _______
---'   ____)____
          ______)
          _______)
         _______)
---.__________)
'''

scissors = '''
    _______
---'   ____)____
          ______)
       __________)
      (____)
---.__(___)
'''
import random
choice=int(input('What do you choose? Type 0 for Rock,1 for Paper or 2 for Scissors'))
if choice==0:
    print(rock)
elif choice==1:
    print(paper)
elif choice==2:
    print(scissors)
else:
    print("Wrong Choice")
computer_choice=random.randint(0,2)
print('Computer\'s Choice.')
if computer_choice==0:
    print(rock)
elif computer_choice==1:
    print(paper)
elif computer_choice==2:
    print(scissors)
if(computer_choice==choice):
    print('It\'s a draw.')
elif choice==0:
    if computer_choice==1:
        print('You loose.')
    else:
        print('You win.')
elif choice==1:
    if computer_choice==0:
        print('You Win.')
    else:
        print('You Loose.')
elif choice==2:
    if computer_choice==2:
        print('You Loose')
    else:
        print('You Win!')