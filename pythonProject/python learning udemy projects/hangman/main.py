import hangman_art
import random
import hangman_words
logo=hangman_art.logo
print(logo)
word=random.choice(hangman_words.word_list)
chances=1
display_string=[]
while chances <7:
    letter=input("Guess a letter: ").lower()
    if letter not in word:
        print(f"You guesses {letter}, that's not in the word. You lose a life")
        print(hangman_art.stages[6-chances])
        chances=chances+1
    else:
        for i in  word:
            if letter != i:
                print("- ")
            else:
                print(letter)
        print(hangman_art.stages[7-chances])



