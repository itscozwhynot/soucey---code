# data array handling
import numpy as np

class SUNS_VOC_PARSE:
    def __init__(self, _file_path, _file_name):

        #build file path
        self.file_path = r'{}\{}'.format(_file_path,_file_name)

        #create dictionary of headers and their respective data
        self.headers = {}

        #import lines from file
        with open(self.file_path, 'r', encoding = 'iso-8859-1') as file:

            lines = file.readlines()

            #extract headers
            headers = np.array(lines[0].strip().replace('Pseudo IV:', '').split('\t'))

            #extract data and place them into individual arrays
            data =  np.column_stack([ np.array(lines[i].strip().split('\t')).astype(np.float32) for i in range(1,len(lines))])
            
            #create a dictionary item for each header and assign the respective data value to it
            for i in range(0,len(headers)):
                self.headers[str(headers[i])] = data[i]
        
        if 'V' in self.headers:
            self.voltage = self.headers['V']
        
        if 'I' in self.headers:
            self.current = self.headers['I']


        
