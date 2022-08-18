import pickle

with open("registered_people.dat" , "wb") as f:
    try:
        print(pickle.load(f))
    except:
        print("nothing")
    # pickle.dump({},f)

