import numpy as np



_files = {
    '.lgt': 'light',
    '.drk': 'dark',
    '.jv': 'iv',
    '.sj': 'isc',
}


class LOANA_PARSE:
    '''
    .lgt: light
    .drk: dark
    .jv: iv
    .sj: isc

    '''
    def __init__(self, _file_path, _file_name):

        # init measurement file type data dict
        self.headers = {}

        # build file path
        self.file_path = r'{}\{}'.format(_file_path,_file_name)

        # import lines from txt file
        with open(self.file_path, 'r', encoding = 'iso-8859-1') as file:
            lines = file.readlines()

        # get line index for data headers
        k = [ i for i in range(len(lines)) if '[Measurement]' in lines[i] ][0]

        # get line index of data
        j = [ i for i in range(len(lines)) if '**Data**' in lines[i] ][0]

        # extract data headers
        _headers = [ lines[i].strip().split('\t')[1] for i in range(k+1, j-1) ]

        # extract data
        _data = np.stack([ np.array(lines[i].strip().split('\t')).astype(np.float32) for i in range(j+1, len(lines)-1) ])

        # store each data array in dict by header
        for i in range(len(_headers)):
            self.headers[_headers[i]] = _data[:, i]

        if 'Voltage' in self.headers:
            self.voltage = self.headers['Voltage']
        
        if 'Current' in self.headers:
            self.current = self.headers['Current']


