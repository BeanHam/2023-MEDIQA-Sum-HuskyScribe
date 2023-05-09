#---------------------------------
# topic ontology for headers
#---------------------------------
topic_ontology = 'Topic categories: GYNECOLOGICAL HISTORY | MEDICATIONS | CHIEF COMPLAINTS | PAST MEDICAL HISTORY | ALLERGY | FAMILY AND SOCIAL HISTORY | PAST SURGICAL | OTHER_HISTORY | ASSESSMENT | REVIEW OF SYSTEM | DISPOSITION | EXAM | PLAN | DIAGNOSIS | ED COURSE | IMMUNIZATIONS | LABS | IMAGING | PROCEDURES | GYNHX'
topic_ontology_new = 'Topic categories: GENERAL HISTORY | MEDICATIONS | CHIEF COMPLAINTS | PAST MEDICAL HISTORY | ALLERGY | FAMILY AND SOCIAL HISTORY | PAST SURGICAL | OTHER_HISTORY | ASSESSMENT | REVIEW OF SYSTEM | DISPOSITION | EXAM | PLAN | DIAGNOSIS | ED COURSE | IMMUNIZATIONS | LABS | IMAGING | PROCEDURES | GYNECOLOGICAL HISTORY'
topic_ontology_canonical = 'Topic categories: CHIEF COMPLAINT | RESULTS | REVIEW OF SYSTEMS | HISTORY OF PRESENT ILLNESS | PHYSICAL EXAM | INSTRUCTIONS | ASSESSMENT AND PLAN | VITALS | MEDICATIONS | MEDICAL HISTORY | FAMILY AND SOCIAL HISTORY | SURGICAL HISTORY | ALLERGIES'

#---------------------------------
# canonical headers
#---------------------------------
CANONICAL_CLASSES = ['CHIEF COMPLAINT',
                     'RESULTS',
                     'REVIEW OF SYSTEMS',
                     'HISTORY OF PRESENT ILLNESS',
                     'PHYSICAL EXAM',
                     'ASSESSMENT AND PLAN',
                     'VITALS',
                     'MEDICATIONS',
                     'MEDICAL HISTORY',
                     'FAMILY AND SOCIAL HISTORY',
                     'SURGICAL HISTORY',
                     'ALLERGIES'
]
CLINICAL_T5_LARGE_MODEL_NAME = '/home/sitongz/UnifiedSKG/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Large'
CLINICAL_T5_SCRATCH_MODEL_NAME = '/home/sitongz/UnifiedSKG/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Scratch'
CLINICAL_T5_SCRATCH_MODEL_NAME_SAFE = 'download_models/Clinical-T5-Scratch'

#--------------------------------------------------------
# Map between taskA headers to taskB cannonical headers
#--------------------------------------------------------
TASKA_TO_CANONICAL = {}
TASKA_TO_CANONICAL['CHIEF COMPLAINTS'] = 'CHIEF COMPLAINT'
TASKA_TO_CANONICAL['GENERAL HISTORY'] = 'HISTORY OF PRESENT ILLNESS'
TASKA_TO_CANONICAL['OTHER_HISTORY'] = 'HISTORY OF PRESENT ILLNESS'
TASKA_TO_CANONICAL['REVIEW OF SYSTEM'] = 'REVIEW OF SYSTEMS'
TASKA_TO_CANONICAL['PAST MEDICAL HISTORY'] = 'MEDICAL HISTORY'
TASKA_TO_CANONICAL['FAMILY AND SOCIAL HISTORY'] = 'FAMILY AND SOCIAL HISTORY'
TASKA_TO_CANONICAL['PAST SURGICAL'] = 'SURGICAL HISTORY'
TASKA_TO_CANONICAL['ALLERGY'] = 'ALLERGIES'
TASKA_TO_CANONICAL['MEDICATIONS'] = 'MEDICATIONS'
TASKA_TO_CANONICAL['IMMUNIZATIONS'] = 'MEDICAL HISTORY'
TASKA_TO_CANONICAL['PROCEDURES'] = 'ASSESSMENT AND PLAN'
TASKA_TO_CANONICAL['GYNECOLOGICAL HISTORY'] = 'HISTORY OF PRESENT ILLNESS'
TASKA_TO_CANONICAL['EXAM'] = 'PHYSICAL EXAM'
TASKA_TO_CANONICAL['DIAGNOSIS'] = 'ASSESSMENT AND PLAN'
TASKA_TO_CANONICAL['LABS'] = 'RESULTS'
TASKA_TO_CANONICAL['IMAGING'] = 'RESULTS'
TASKA_TO_CANONICAL['ASSESSMENT'] = 'ASSESSMENT AND PLAN'
TASKA_TO_CANONICAL['PLAN'] = 'ASSESSMENT AND PLAN'
TASKA_TO_CANONICAL['OTHER'] = 'OTHER'
TASKA_TO_CANONICAL['DISPOSITION'] = 'ASSESSMENT AND PLAN'
TASKA_TO_CANONICAL['ED COURSE'] = 'HISTORY OF PRESENT ILLNESS'

#--------------------------------------------------------
# abbreviated header to full header
#--------------------------------------------------------
SECTION2FULL = {'GENHX': 'General History',
                'FAM/SOCHX': 'Family and Social History',
                'PASTSURGICAL': 'past surgical',
                'ROS': 'Review of System',
                'ASSESSMENT': 'Assessment',
                'PASTMEDICALHX': 'Past Medical History',
                'MEDICATIONS': 'MEDICATIONS',
                'GYNHX': 'gynecological history',
                'CC': 'Chief Complaints',
                'EDCOURSE': 'emergency department'
}

#--------------------------------------------
# Subsections for each first-level section
#--------------------------------------------
SUBJECTIVE = ['CHIEF COMPLAINT', 
              'HISTORY OF PRESENT ILLNESS', 
              'REVIEW OF SYSTEMS', 
              'MEDICAL HISTORY', 
              'MEDICATIONS',
              'FAMILY AND SOCIAL HISTORY',
              'SURGICAL HISTORY',
              'ALLERGIES']
OBJECTIVE_EXAM = ['PHYSICAL EXAM']
OBJECTIVE_RESULTS = ['RESULTS', 'VITALS']
AP = ['ASSESSMENT AND PLAN', 'INSTRUCTIONS']

#----------------------------------------------
# Collection of section and subsection headings
#----------------------------------------------
SECTIONS = {}
SECTIONS['SUBJECTIVE'] = SUBJECTIVE
SECTIONS['OBJECTIVE_EXAM'] = OBJECTIVE_EXAM
SECTIONS['OBJECTIVE_RESULTS'] = OBJECTIVE_RESULTS
SECTIONS['ASSSESSMENT'] = AP
