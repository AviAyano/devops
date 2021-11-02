import collections
import re
MAX_TRIES = 6
    
def hangman_photos(situation_number):
  """ exp: 
            The progream print the hangman positions
        tayps:
            number - int
        return:
            None
  """
  number = int(situation_number)
  if number == -1:
    print(""" 
                    _    _
                   | |  | |
                   | |__| | __ _ _ __   __ _ _ __ ___   __ _ _ __
                   |  __  |/ _' | '_ \\ / _' | '_ ' _ \\ / _' | '_ \\
                   | |  | | (_| | | | | (_| | | | | | | (_| | | | |
                   |_|  |_|\__,_|_| |_|\\__, |_| |_| |_|\\__,_|_| |_|
                                        __/ |
                                       |___/ \n """
                   ,MAX_TRIES )
     
  elif number == 0:
    print( """
    x-------x
    """)
  elif number == 1:
    print( """
    x-------x
    |
    |
    |
    |
    | 
        """)

  elif number == 2:
    print( """
    x-------x
    |       |
    |       0
    |
    |
    |

        """)
  elif number == 3:
    print( """
    x-------x
    |       |
    |       0
    |       |
    |
    |
        """) 

  elif number == 4:
    print( """
    x-------x
    |       |
    |       0
    |      /|\\
    |
    |
        """)

  elif number == 5:
    print( """
    x-------x
    |       |
    |       0
    |      /|\\
    |      /
    |
        """)

  elif number == 6:
    print( """
    x-------x
    |       |
    |       0
    |      /|\\
    |      / \\
    |
        """)
  else:
    print("Error")
  
  
def show_hidden_word(secret_word, old_letters_guessed):
    """ exp: 
            The progream show the secret word that exposed by the user
        tayps:
            old_letters_guessed - list
            game_status_string,secret_word - string
        return:
            True/False (boolean) 
    """
    game_status_string = " "
    for letter in secret_word:
        if letter in old_letters_guessed:
            game_status_string = game_status_string + " " + letter
        else: 
            game_status_string = game_status_string + " " + "___"
    return game_status_string  
  

def choose_word(file_path, index):
    """ exp: 
            progream that choose the secret word from the pahe file
        tayps:
            files_path - function that read from the file
            line,words - list
            words_from_line,word - string
            duplicate_of_words,size,input_number - int
            words_from_line - string
            
        return:
            word (string) = a secret word from the file
    """
    input_number = int(index)
    files_path = open (file_path, "r")
    words_from_line = " "
    for checking_word in files_path:
       line = checking_word.split(" ")
       for word_in in range(len(line)):
            words_from_line = words_from_line+ " " + line[word_in]
       words = words_from_line.split()
       duplicate_of_words = len([item for item, count in collections.Counter(words).items() if count > 1])
    size = len(words)
    if input_number > size: 
        input_number = input_number % size
    word = words[(input_number - 1)]       
    files_path.close()
    return word
 
 
def check_win(secret_word, old_letters_guessed):
    """ exp: 
            progream that check if the secret word is in the list of old_letters_guessed
        tayps:
            old_letters_guessed - list
            secret_word - string
            letters - list
        return:
            True/False (boolean) 
    """
    letters = list(secret_word)
    checking_win = 0
    len_of_secret_word = len(secret_word)
    for letter in letters:
        if letter not in old_letters_guessed:
            return False
        else:
            checking_win += 1
            if checking_win == len_of_secret_word:
                return True 


def check_valid_input(letter_guessed, old_letters_guessed):
    """ exp: 
            The progream check input validation
        tayps:
            old_letters_guessed - list
            correct_input,letter_guessed - string
        return:
            True/False (boolean) 
    """
    old_letters_guessed.sort()
    if len(letter_guessed) == 1:
        if letter_guessed not in old_letters_guessed:
            return True
        else : 
            return False
    elif len(letter_guessed) > 1:
        return False    
       

def try_update_letter_guessed(letter_guessed, old_letters_guessed):
    """ exp: 
            The progream check input validation
        tayps:
             None
        return:
            True/False (boolean) 
    """
    return check_valid_input(letter_guessed, old_letters_guessed)
    
def main():
    finish_the_game_flag = True
    num_of_tries = 0
    HANGMAN_PHOTOS = {}
    hangman_photos(-1)
    old_letters_guessed = []
    regex = re.compile('[1234567890@_!#$%^&*()<>?/\|}{~:]')
    file_path = input("\n Please enter a file path: ")
    number = input("\n Please enter index: ")
    print("\n Let's start ! \n")
    
    secret_word = choose_word(file_path, number)
    hangman_photos(num_of_tries)
    show_hidde_word = show_hidden_word(secret_word, old_letters_guessed)
    print('\n', show_hidde_word, '\n')
    
    while finish_the_game_flag:
        letter_guessed = input("Please guess a letter: ")     
        old_letters_guessed.sort()

        if(regex.search(letter_guessed) == None):
        
            if(letter_guessed not in old_letters_guessed) or (letter_guessed.lower() not in old_letters_guessed):
                
                if (letter_guessed in secret_word) or (letter_guessed.lower() in secret_word):
                
                    if try_update_letter_guessed(letter_guessed , old_letters_guessed):
                        old_letters_guessed += letter_guessed.lower()
                        old_letters_guessed += letter_guessed             
                        show_hidde_word = show_hidden_word(secret_word, old_letters_guessed)
                        print('\n', show_hidde_word, '\n')
                        
                        if check_win(secret_word, old_letters_guessed):
                            print("\n WIN ! " )
                            finish_the_game_flag = False
               
                else:
                    print("\n ): \n")
                    num_of_tries += 1
                    hangman_photos(num_of_tries)
                    show_hidde_word = show_hidden_word(secret_word, old_letters_guessed)
                    print('\n', show_hidde_word, '\n')
                    
                    if num_of_tries == MAX_TRIES: 
                        print("\n Lose ! " )
                        finish_the_game_flag = False
            else:
                print("X")
                print(*old_letters_guessed,sep = " -> " )
        else:
            print("X")
            print(*old_letters_guessed,sep = " -> " )   
    
if __name__ == "__main__":
    main()