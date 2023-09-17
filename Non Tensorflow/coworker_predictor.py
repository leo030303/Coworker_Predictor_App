import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from joblib import dump, load
stay_length_predictor = load('stay_length_predictor.joblib')
quit_predictor = load('quit_predictor.joblib')
def datacleaner(coworker_age, coworker_start_month, coworker_education, coworker_english, coworker_gender, coworker_experience):
    print("Cleaning Data...")
    month1 = 1 if coworker_start_month in ["4"] else 0
    month2 = 1 if coworker_start_month in ["8"] else 0
    month3 = 1 if coworker_start_month in ["7"] else 0
    month4 = 1 if coworker_start_month in ["3", "2", "1"] else 0
    month5 = 1 if coworker_start_month in ["5"] else 0
    month6 = 1 if coworker_start_month in ["11", "12"] else 0
    month7 = 1 if coworker_start_month in ["10","9"] else 0
    month8 = 1 if coworker_start_month in ["6"] else 0
    age1 = 1 if coworker_age in range(10,20) else 0
    age2 = 1 if coworker_age in range(20,22) else 0
    age3 = 1 if coworker_age in range(22,24) else 0
    age4 = 1 if coworker_age in range(24,26) else 0
    age5 = 1 if coworker_age in range(26,28) else 0
    age6 = 1 if coworker_age in range(28,35) else 0
    ed1 = 1 if coworker_education == "1" else 0
    ed2 = 1 if coworker_education == "2" else 0
    en1 = 1 if coworker_english == "3" else 0
    en2 = 1 if coworker_english == "2" else 0
    gen = 1 if coworker_gender == "2" else 0
    ex1 = 1 if coworker_experience == "1" else 0
    ex2 = 1 if coworker_experience == "2" else 0
    data = {"Start Month_April":[month1],
    "Start Month_August":[month2],
    "Start Month_July":[month3],
    "Start Month_June":[month8],
    "Start Month_March":[month4],
    "Start Month_May":[month5],
    "Start Month_November":[month6],
    "Start Month_October":[month7],
    "Age_(10, 20]":[age1],
    "Age_(20, 22]":[age2],
    "Age_(22, 24]":[age3],
    "Age_(24, 26]":[age4],
    "Age_(26, 28]":[age5],
    "Age_(28, 35]":[age6],
    "Education_College":[ed1],
    "Education_Secondary":[ed2],
    "English level_Fluent":[en1],
    "English level_Intermediate":[en2],
    "Gender_Female":[gen],
    "Bar Experience_Bar":[ex1],
    "Bar Experience_Waiter":[ex2]}
    df = pd.DataFrame(data)
    print("Data Cleaned.")
    return df
def datacollector():
    coworker_name = input("Whats the new victims name: ")
    coworker_age = input(f"How old is {coworker_name}: ")
    while not coworker_age.isdigit():
        coworker_age = input(f"Come on give me a real age, I know what you're at, how old is {coworker_name}: ")
    coworker_age = int(coworker_age)
    if coworker_age <= 15:
        print(f"Christ I knew the Reidys were bad, but child labour? Really? Poor {coworker_name}. Call the guards please")
        return [False, coworker_name, coworker_age, "blank", "blank", "blank", "blank", "blank"]
    elif coworker_age >= 35:
        print(f"Sorry, model can't go that high, maybe tell {coworker_name} to apply for the pension or something.")
        return [False, coworker_name, coworker_age, "blank", "blank", "blank", "blank", "blank"]
    else:
        coworker_start_month = input(f"Alright, which month did poor {coworker_name} start in (1-12): ")
        while not coworker_start_month in ['1','2','3','4','5','6','7','8','9','10','11','12']:
            coworker_start_month = input("Learn to read you little bollocks, number between 1 and 12: ")
        coworker_education = input(f"Is {coworker_name} (1)Failing college, (2)Shitting themselves for the LC or (3)A waster: ")
        while not coworker_education in ['1','2','3']:
            coworker_education = input(f"Well you clearly can't read, is {coworker_name} (1)Failing college, (2)Shitting themselves for the LC or (3)A waster: ")
        coworker_english = input(f"An bhfuil {coworker_name} abalta a laibhairt as Bearla (1)go dona, (2)go maith no (3)go liofa: ")
        while not coworker_english in ['1','2','3']:
            coworker_english = input(f"Is ejit thu, aris, an bhfuil {coworker_name} abalta a laibhairt as Bearla (1)go dona, (2)go maith no (3)go liofa: ")
        coworker_gender = input("Are they a (1)man or a (2)woman (I know they aren't non binary, the Reidys would never hire one of those): ")
        while not coworker_gender in ["1","2"]:
            coworker_gender = input("Come on, give me a 1 or a 2, are they a (1)man or a (2)woman: ")
        coworker_pronoun = "she"
        if coworker_gender == "1":
            coworker_pronoun = "he"
        coworker_experience = input(f"And lastly, does {coworker_pronoun} have (1)Bar experience, (2)Waiter experience, or (3)Neither: ")
        while not coworker_experience in ['1','2','3']:
            coworker_experience = input(f"Come on, don't let your literacy fail you now, does {coworker_pronoun} have (1)Bar experience, (2)Waiter experience, or (3)Neither: ")
        print("\nAlrighty, got the info, lemme just process this")
        return [True, coworker_name, coworker_age, coworker_start_month, coworker_education, coworker_english, coworker_gender, coworker_experience]

def main():
    run_system = True
    while run_system:
        [do_next, coworker_name, coworker_age, coworker_start_month, coworker_education, coworker_english, coworker_gender, coworker_experience] = datacollector()
        if do_next:
            df = datacleaner(coworker_age, coworker_start_month, coworker_education, coworker_english, coworker_gender, coworker_experience)
            stay_val = stay_length_predictor.predict(df)
            quit_val = quit_predictor.predict(df)
            print(f"{coworker_name} will endure Le Shithole for {stay_val[0]} days, god bless their soul.")
        run_val = input("Enter y if you want to run another test, or press any other key to quit: ")
        if run_val not in ["y", "Y"]:
            run_system = False
    print("Slan go foil")


main()
