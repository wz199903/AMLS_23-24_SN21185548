from os import system as run
import sys


# Task A switch
def switch_pneumonia():
    def Adata():
        run(f"python {'A/data_preprocessing_pneumonia.py'}")

    def Atrain():
        run(f"python {'A/train_pneumonia.py'}")

    def Atest():
        run(f"python {'A/evaluation_pneumonia.py'}")

    def Adefault():
        print("Please enter a valid option.")
        switch_pneumonia()

    # User input
    try:
        option = int(input("Enter 1 for data preprocessing and balance class occurrence\nEnter 2 for model training\n"
                           "Enter 3 for model testing: "))
    except ValueError:
        Adefault()
        return

    switch_dict = {
        1: Adata,
        2: Atrain,
        3: Atest,
    }

    switch_dict.get(option, Adefault)()


# Task B switch
def switch_path():

    def Bdata():
        run(f"python {'B/data_preprocessing_path.py'}")

    def Bbase_train():
        run(f"python {'B/train_path.py'}")

    def Bspecialised_train():
        run(f"python {'B/specialised_path.py'}")

    def Btest():
        run(f"python {'B/evaluation_path.py'}")

    def Bdefault():
        print("Please enter a valid option.")
        switch_path()

    # User input
    option = int(
        input(
            "Enter 1 for data preprocessing and balance class occurrence\nEnter 2 for base model training\n"
            "Enter 3 for specialised model training\nEnter 4 for model testing: "))

    switch_dict = {
        1: Bdata,
        2: Bbase_train,
        3: Bspecialised_train,
        4: Btest,
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