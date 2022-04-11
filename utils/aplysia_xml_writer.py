"""
        input: .xml file with the skeleton
        output: .xml filled out with the desired params

"""

#TODO: make input path, output path, params_values readable
# from the outside as args

if __name__ == '__main__':
    import xml.etree.ElementTree as ET
    import random



    input_path = 'skeleton.xml'
    output_path = 'adult20years.xml'

    # set here time interval values
    time_interval = 20

    # set here desired values
    params_values = {'HR': None,
                     'TotalVascularVolume': None,
                     'e_lvmax': None,
                     'e0_lv': None,
                     'e_rvmax': None,
                     'e0_rv': None,
                     'SVR': None,
                     'PVR': None,
                     'Ea': None,
                     'Epa': None,
                     'Hb': None,
                     'O2Cons': None,
                     'PulmShuntFraction': None,
                     'p_low': None}


    # open the skeleton
    skeleton = ET.parse(input_path)
    root = skeleton.getroot()

    # set the time interval
    root.find('TimeInterval')[0].text = str(time_interval)

    # update with the list of params
    for parameter in root.iter('Parameter'):
        param_name = parameter[1].text
        values = params_values[param_name]
        for value in values:
            new_value = ET.SubElement(parameter, "Value")
            new_value.text = str(value)

    skeleton.write(output_path)