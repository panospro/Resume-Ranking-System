EDUCATION_LEVELS = {
    'PhD': r'\b(Ph\.?D\.?|Doctor of Philosophy)\b',
    'Masters': r'\b(M\.?Sc\.?|M\.?Tech\.?|M\.?E\.?|M\.?Eng\.?|MBA|Master[’\'s]* of [A-Za-z]+)\b',
    'Bachelors': r'\b(B\.?Sc\.?|B\.?Tech\.?|B\.?E\.?|B\.?Eng\.?|B\.?Com\.?|B\.?C\.?A\.?|B\.?B\.?A\.?|Bachelor[’\'s]* of [A-Za-z]+)\b',
    'Diploma': r'\b(Diploma|Polytechnic|Associate Degree)\b',
    'High School': r'\b(High School|12th|10th|Secondary|Senior Secondary|HSC|SSC)\b'
}

EDU_RANK = {
    "PhD": 5,
    "Masters": 4,
    "MBA": 4,
    "Bachelor's": 3,
    "Bachelors": 3,
    "Possibly Bachelors": 3,
    "Diploma": 2,
    "High School": 1,
    "Unknown": 0,
    "Not Specified": 0
}