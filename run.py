import os

data_name = "KG2ID_BRCA_STRING_CV_05_01/"
print(data_name)

if not os.path.exists('./Output/'):
    os.mkdir('./Output/')

os.system("python processDynMap.py -d " + data_name)
print("Done.")

