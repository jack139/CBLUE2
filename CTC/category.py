
category = [
    'Disease',
    'Symptom',
    'Sign',
    'Pregnancy-related Activity',
    'Neoplasm Status',
    'Non-Neoplasm Disease Stage',
    'Allergy Intolerance',
    'Organ or Tissue Status',
    'Life Expectancy',
    'Oral related',
    'Pharmaceutical Substance or Drug',
    'Therapy or Surgery',
    'Device',
    'Nursing',
    'Diagnostic',
    'Laboratory Examinations',
    'Risk Assessment',
    'Receptor Status',
    'Age',
    'Special Patient Characteristic',
    'Literacy',
    'Gender',
    'Education',
    'Address',
    'Ethnicity',
    'Consent',
    'Enrollment in other studies',
    'Researcher Decision',
    'Capacity',
    'Ethical Audit',
    'Compliance with Protocol',
    'Addictive Behavior',
    'Bedtime',
    'Exercise',
    'Diet',
    'Alcohol Consumer',
    'Sexual related',
    'Smoking Status',
    'Blood Donation',
    'Encounter',
    'Disabilities',
    'Healthy',
    'Data Accessible',
    'Multiple',
]

category2 = [i.lower() for i in category]

def category_index(s):
    return category2.index(s.lower())

def category_name(i):
    return category[i]
