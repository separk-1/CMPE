import re

def Tool_File_Sort(Data):

    Convert = lambda text: int(text) if text.isdigit() else text.lower()
    Key = lambda key: [ Convert(c) for c in re.split('([0-9]+)', key) ] 
    
    return sorted(Data, key=Key)