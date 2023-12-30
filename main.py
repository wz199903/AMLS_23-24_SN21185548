from os import system as run
from os.path import join
import sys

# Import relevant files
sys.path.insert(0, 'A')
import xxx
import xxx
import xxx
sys.path.insert(0, 'B')
import xxx
import xxx
import xxx


# Task A switch
def switch_pneumonia():
    def Acv():
        xxx.run()

    def Atrain():
        xxx_train.run()

    def Atest():
        xxx.run()

    def Adefault():
        print("Please enter a valid option.")
        switch_pneumonia()

    # User input
    option = int(
        input(
            "Enter 1 cross-validation\nEnter 2 for model training\nEnter 3 for model testing:\n"))

    switch_dict = {
        1: Acv,
        2: Atrain,
        3: Atest,
    }

    switch_dict.get(option, Adefault)()

# Task B switch
def switch_path():

    def Bcv():
        xxx.run()

    def Btrain():
        xxx.run()

    def Btest():
        xxx.run()

    def Bdefault():
        print("Please enter a valid option.")
        switch_path()

    # User input
    option = int(
        input(
            "Enter 1 cross-validation\nEnter 2 for model training\nEnter 3 for model testing:\n"))

    switch_dict = {
        1: Bcv,
        2: Btrain,
        3: Btest,
    }

    switch_dict.get(option, Bdefault)()


# Main switch
def switch_main():
    # Task A
    def A():
        PneumoniaMNIST()

    # Task B
    def B():
        PathMNIST()

    def default():
        print("Please enter a valid option.")
        switch_main()

    def quit_main():
        quit()

    # User input
    option = int(
        input(
            "Enter 1 for task A\nEnter 2 for task B\nEnter 3 to exit program:\n"))

    switch_dict = {
        1: PneumoniaMNIST,
        2: PathMNIST,
        3: quit_main,
    }

    switch_dict.get(option, default)()

# Call switch
switch_main()
quit()