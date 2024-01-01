from os import system as run
from os.path import join
import sys

path_to_A = 'A'
path_to_B = 'B'

# Import relevant files
sys.path.insert(0, 'A')
#from import model_pneumonia
#import train_pneumonia
#import evaluation_pneumonia
#sys.path.insert(0, 'B')
#import model_path
#import train_path
#import evaluation_path


# Task A switch
def switch_pneumonia():
    def Acv():
        run(f"python {join(path_to_A, 'data_preprocessing_pneumonia.py')}")

    def Atrain():
        run(f"python {join(path_to_A, 'train_pneumonia.py')}")

    def Atest():
        run(f"python {join(path_to_A, 'evaluation_pneumonia.py')}")

    def Adefault():
        print("Please enter a valid option.")
        switch_pneumonia()

    # User input
    try:
        option = int(input("Enter 1 for cross-validation\nEnter 2 for model training\nEnter 3 for model testing: "))
    except ValueError:
        Adefault()
        return

    switch_dict = {
        1: Acv,
        2: Atrain,
        3: Atest,
    }

    switch_dict.get(option, Adefault)()

# Task B switch
def switch_path():

    def Bcv():
        run(f"python {join(path_to_A, 'data_preprocessing_path.py')}")

    def Btrain():
        run(f"python {join(path_to_A, 'train_path.py')}")

    def Btest():
        run(f"python {join(path_to_A, 'evaluation_path.py')}")

    def Bdefault():
        print("Please enter a valid option.")
        switch_path()

    # User input
    option = int(
        input(
            "Enter 1 cross-validation\nEnter 2 for model training\nEnter 3 for model testing: "))

    switch_dict = {
        1: Bcv,
        2: Btrain,
        3: Btest,
    }

    switch_dict.get(option, Bdefault)()


# Main switch
def switch_main():
    # Task A
    def a():
        switch_pneumonia()

    # Task B
    def b():
        switch_path()

    def default():
        print("Please enter a valid option.")
        switch_main()

    def quit_main():
        sys.exit()

    try:
        option = int(input("Enter 1 for task A\nEnter 2 for task B\nEnter 3 to exit program: "))
    except ValueError:
        default()
        return

    switch_dict = {
        1: a,
        2: b,
        3: quit_main,
    }

    switch_dict.get(option, default)()


# Call switch
switch_main()