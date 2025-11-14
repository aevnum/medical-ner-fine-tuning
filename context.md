# Medical Entity Recognition and Entity Linking

## MACCROBAT Dataset

### Loading
```python
# load the data
dataset = load_dataset("ktgiahieu/maccrobat2018_2020")
print(dataset)

# Check structure
print(f"Train examples: {len(dataset['train'])}")
print(f"Example structure:")
example = dataset['train'][0]
print(f"  Keys: {example.keys()}")
print(f"  Tokens: {example['tokens'][:10]}")
print(f"  Tags: {example['tags'][:10]}")

# Get unique tags for num_labels
unique_tags = set()
for example in dataset['train']:
    unique_tags.update(example['tags'])

# Ensure every B-XXX tag has a corresponding I-XXX tag
extra_i_tags = set()
for tag in unique_tags:
    if tag.startswith("B-"):
        i_tag = "I-" + tag[2:]
        if i_tag not in unique_tags:
            extra_i_tags.add(i_tag)
unique_tags.update(extra_i_tags)

unique_tags = sorted(list(unique_tags))
num_labels = len(unique_tags)
tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}
```

### Dataset Information
#### Sample Data
Tokens: ['In', 'January', '2009', ',', 'a', '57', '-', 'year', '-', 'old', 'woman', 'emergently', 'presented', 'with', 'acute', 'shortness', 'of', 'breath', '.', '\n', 'She', 'had', 'a', '1', '-', 'month', 'history', 'of', 'progressive', 'shortness', 'of', 'breath', 'and', 'a', 'gradual', 'decrease', 'in', 'exercise', 'capacity', 'secondary', 'to', 'mild', 'dyspnea', '.', '\n', 'She', 'reported', 'no', 'additional', 'symptoms', '.', '\n', 'At', 'age', '40', ',', 'she', 'had', 'been', 'diagnosed', 'with', 'a', 'stage', 'IIA', ',', 'T1bN1', ',', 'left', '-', 'sided', 'breast', 'cancer', '.', '\n', 'Initial', 'treatment', 'had', 'included', 'a', 'lumpectomy', 'and', 'axillary', 'node', 'dissection', '.', '\n', 'She', 'subsequently', 'underwent', '4', 'cycles', 'of', 'DOX', 'therapy', '(', '75', 'mg', '/', 'm2', ')', ',', 'followed', 'by', '8', 'cycles', 'of', 'cyclophosphamide', ',', 'methotrexate', ',', 'and', '5', '-', 'fluorouracil', '.', '\n', 'Multigated', 'acquisition', 'scans', 'before', 'and', 'after', 'chemotherapy', 'showed', 'normal', 'cardiac', 'function', '.', '\n', 'After', 'chemotherapy', ',', 'she', 'underwent', 'left', 'whole', '-', 'breast', 'radiation', 'with', 'an', 'axillary', 'boost', '.', '\n', 'Because', 'the', 'tumor', 'had', 'been', 'estrogen', 'receptor', '-', 'positive', ',', 'her', 'subsequent', 'medical', 'regimen', 'consisted', 'only', 'of', 'anti', '-', 'estrogen', 'therapy', '.', '\n', 'She', 'took', 'tamoxifen', 'for', '5', 'years', ',', 'and', ',', 'ever', 'since', ',', 'the', 'aromatase', 'inhibitor', 'letrozole', '.', '\n', 'In', 'the', '17', 'years', 'after', 'chemotherapy', ',', 'she', 'had', 'been', 'active', 'and', 'in', 'relatively', 'good', 'health', '.', '\n', 'In', 'addition', 'to', 'her', 'other', 'symptoms', ',', 'she', 'now', 'presented', 'with', 'tachycardia', ',', 'tachypnea', ',', 'and', 'hypertension', '.', '\n', 'She', 'had', 'marked', 'jugular', 'venous', 'distention', ',', 'an', 'S3', ',', 'pulmonary', 'rales', ',', 'and', 'trace', 'peripheral', 'edema', '.', '\n', 'Initial', 'laboratory', 'values', 'were', 'within', 'normal', 'limits', 'except', 'for', 'an', 'elevated', 'level', 'of', 'N', '-', 'terminal', 'pro', '-', 'brain', 'natriuretic', 'peptide', '(', '>', '2,000', 'pg', '/', 'mL', ')', '.', '\n', 'Results', 'of', 'investigation', 'into', 'the', 'new', '-', 'onset', 'cardiomyopathy', 'included', 'normal', 'cardiac', 'enzyme', 'levels', ',', 'an', 'electrocardiogram', '(', 'ECG', ')', 'that', 'revealed', 'no', 'ischemic', 'changes', ',', 'and', 'a', 'coronary', 'angiogram', 'of', 'normal', 'appearance', '.', '\n', 'The', 'ECG', 'showed', 'sinus', 'tachycardia', 'with', 'frequent', 'premature', 'ventricular', 'complexes', ',', 'left', '-', 'axis', 'deviation', ',', 'left', 'atrial', 'enlargement', ',', 'and', 'low', '-', 'voltage', 'QRS', 'complexes', 'with', 'nonspecific', 'ST', 'changes', '(', 'Fig.1', ')', '.', '\n', 'A', '2', '-', 'dimensional', 'echocardiogram', 'revealed', 'a', 'left', 'ventricular', 'ejection', 'fraction', '(', 'LVEF', ')', 'of', '0.20', ',', 'severe', 'diffuse', 'left', 'ventricular', '(', 'LV', ')', 'hypokinesis', ',', 'and', 'a', 'mildly', 'dilated', 'left', 'atrium', '.', '\n', 'To', 'better', 'define', 'the', 'cause', 'of', 'the', 'LV', 'systolic', 'dysfunction', ',', 'cardiovascular', 'magnetic', 'resonance', '(', 'CMR', ')', 'was', 'performed', '.', '\n', 'It', 'confirmed', 'the', 'LVEF', 'of', '0.20', '.', '\n', 'The', 'T2', '-', 'weighted', 'sequence', 'showed', 'slow', 'flow', 'secondary', 'to', 'LV', 'dysfunction', ',', 'and', 'no', 'myocardial', 'edema', '(', 'Fig.2A', ')', '.', '\n', 'Late', 'gadolinium', 'enhancement', 'disclosed', 'diffuse', 'myocardial', 'thinning', 'and', 'no', 'scarring', '(', 'Fig.2B', ')', '.', '\n', 'The', 'patient', 'was', 'treated', 'medically', '.', '\n', 'Her', 'symptoms', 'progressively', 'improved', 'during', 'therapy', ',', 'which', 'consisted', 'of', 'a', 'β', '-', 'blocker', ',', 'an', 'angiotensin', '-', 'converting', 'enzyme', 'inhibitor', ',', 'digoxin', ',', 'and', 'a', 'diuretic', '.', '\n', 'The', 'therapy', 'was', 'slowly', 'tapered', ',', 'and', 'her', 'LVEF', 'increased', 'from', '0.20', 'to', '0.55', 'during', 'an', '8', '-', 'month', 'period', '.', '\n', 'All', 'medications', 'except', 'for', 'low', '-', 'dose', 'metoprolol', 'were', 'discontinued', 'after', '1', 'year', ',', 'and', 'she', 'remained', 'asymptomatic', '.']
Tags: ['O', 'B-Date', 'I-Date', 'O', 'O', 'B-Age', 'I-Age', 'I-Age', 'I-Age', 'I-Age', 'B-Sex', 'O', 'B-Clinical_event', 'O', 'O', 'B-Sign_symptom', 'I-Sign_symptom', 'I-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'B-Duration', 'I-Duration', 'I-Duration', 'O', 'O', 'O', 'B-Sign_symptom', 'I-Sign_symptom', 'I-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'I-Sign_symptom', 'O', 'O', 'O', 'B-Date', 'I-Date', 'O', 'O', 'O', 'O', 'B-Clinical_event', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Disease_disorder', 'I-Disease_disorder', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Therapeutic_procedure', 'O', 'O', 'B-Therapeutic_procedure', 'I-Therapeutic_procedure', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Medication', 'I-Medication', 'O', 'B-Dosage', 'I-Dosage', 'I-Dosage', 'I-Dosage', 'O', 'O', 'O', 'O', 'B-Dosage', 'I-Dosage', 'O', 'B-Medication', 'O', 'B-Medication', 'O', 'O', 'B-Medication', 'I-Medication', 'I-Medication', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'O', 'O', 'B-Medication', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'O', 'O', 'B-Medication', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Therapeutic_procedure', 'O', 'O', 'B-Detailed_description', 'I-Detailed_description', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'B-Detailed_description', 'I-Detailed_description', 'I-Detailed_description', 'I-Detailed_description', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Medication', 'I-Medication', 'I-Medication', 'I-Medication', 'O', 'O', 'O', 'O', 'B-Medication', 'O', 'B-Duration', 'I-Duration', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Medication', 'O', 'O', 'O', 'O', 'B-Duration', 'I-Duration', 'I-Duration', 'O', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'B-Sign_symptom', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'B-Sign_symptom', 'O', 'B-Sign_symptom', 'I-Sign_symptom', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'B-Lab_value', 'I-Lab_value', 'I-Lab_value', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'B-Lab_value', 'I-Lab_value', 'I-Lab_value', 'I-Lab_value', 'I-Lab_value', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Disease_disorder', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'O', 'B-Diagnostic_procedure', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'I-Sign_symptom', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'O', 'B-Lab_value', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'I-Sign_symptom', 'I-Sign_symptom', 'I-Sign_symptom', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'I-Sign_symptom', 'O', 'O', 'B-Sign_symptom', 'I-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'B-Diagnostic_procedure', 'O', 'O', 'B-Lab_value', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'B-Biological_structure', 'I-Biological_structure', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'B-Diagnostic_procedure', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'O', 'B-Lab_value', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Disease_disorder', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'I-Diagnostic_procedure', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Therapeutic_procedure', 'I-Therapeutic_procedure', 'O', 'O', 'O', 'B-Sign_symptom', 'O', 'B-Lab_value', 'O', 'B-Therapeutic_procedure', 'O', 'O', 'O', 'O', 'O', 'B-Medication', 'I-Medication', 'I-Medication', 'O', 'O', 'B-Medication', 'I-Medication', 'I-Medication', 'I-Medication', 'I-Medication', 'O', 'B-Medication', 'O', 'O', 'O', 'B-Medication', 'O', 'O', 'O', 'B-Coreference', 'O', 'O', 'B-Lab_value', 'O', 'O', 'O', 'B-Diagnostic_procedure', 'B-Lab_value', 'O', 'B-Lab_value', 'O', 'B-Lab_value', 'O', 'O', 'B-Duration', 'I-Duration', 'I-Duration', 'I-Duration', 'O', 'O', 'O', 'B-Medication', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Lab_value', 'B-Date', 'I-Date', 'I-Date', 'O', 'O', 'O', 'O', 'B-Sign_symptom', 'O']
Text:
In January 2009, a 57- year- old woman emergently presented with acute shortness of breath. 
 She had a 1- month history of progressive shortness of breath and a gradual decrease in exercise capacity secondary to mild dyspnea. 
 She reported no additional symptoms. 
 At age 40, she had been diagnosed with a stage IIA, T1bN1, left- sided breast cancer. 
 Initial treatment had included a lumpectomy and axillary node dissection. 
 She subsequently underwent 4 cycles of DOX therapy ( 75 mg / m2), followed by 8 cycles of cyclophosphamide, methotrexate, and 5- fluorouracil. 
 Multigated acquisition scans before and after chemotherapy showed normal cardiac function. 
 After chemotherapy, she underwent left whole- breast radiation with an axillary boost. 
 Because the tumor had been estrogen receptor- positive, her subsequent medical regimen consisted only of anti- estrogen therapy. 
 She took tamoxifen for 5 years, and, ever since, the aromatase inhibitor letrozole. 
 In the 17 years after chemotherapy, she had been active and in relatively good health. 
 In addition to her other symptoms, she now presented with tachycardia, tachypnea, and hypertension. 
 She had marked jugular venous distention, an S3, pulmonary rales, and trace peripheral edema. 
 Initial laboratory values were within normal limits except for an elevated level of N- terminal pro- brain natriuretic peptide ( > 2,000 pg / mL). 
 Results of investigation into the new- onset cardiomyopathy included normal cardiac enzyme levels, an electrocardiogram ( ECG) that revealed no ischemic changes, and a coronary angiogram of normal appearance. 
 The ECG showed sinus tachycardia with frequent premature ventricular complexes, left- axis deviation, left atrial enlargement, and low- voltage QRS complexes with nonspecific ST changes ( Fig.1). 
 A 2- dimensional echocardiogram revealed a left ventricular ejection fraction ( LVEF) of 0.20, severe diffuse left ventricular ( LV) hypokinesis, and a mildly dilated left atrium. 
 To better define the cause of the LV systolic dysfunction, cardiovascular magnetic resonance ( CMR) was performed. 
 It confirmed the LVEF of 0.20. 
 The T2- weighted sequence showed slow flow secondary to LV dysfunction, and no myocardial edema ( Fig.2A). 
 Late gadolinium enhancement disclosed diffuse myocardial thinning and no scarring ( Fig.2B). 
 The patient was treated medically. 
 Her symptoms progressively improved during therapy, which consisted of a β- blocker, an angiotensin- converting enzyme inhibitor, digoxin, and a diuretic. 
 The therapy was slowly tapered, and her LVEF increased from 0.20 to 0.55 during an 8- month period. 
 All medications except for low- dose metoprolol were discontinued after 1 year, and she remained asymptomatic.

Entities:
  • January 2009                             → Date
  • 57 - year - old                          → Age
  • woman                                    → Sex
  • presented                                → Clinical_event
  • shortness of breath                      → Sign_symptom
  • 1 - month                                → Duration
  • shortness of breath                      → Sign_symptom
  • exercise capacity                        → Diagnostic_procedure
  • dyspnea                                  → Sign_symptom
  • additional symptoms                      → Sign_symptom
  • age 40                                   → Date
  • diagnosed                                → Clinical_event
  • breast cancer                            → Disease_disorder
  • lumpectomy                               → Therapeutic_procedure
  • node dissection                          → Therapeutic_procedure
  • DOX therapy                              → Medication
  • 75 mg / m2                               → Dosage
  • 8 cycles                                 → Dosage
  • cyclophosphamide                         → Medication
  • methotrexate                             → Medication
  • 5 - fluorouracil                         → Medication
  • Multigated acquisition scans             → Diagnostic_procedure
  • chemotherapy                             → Medication
  • cardiac function                         → Diagnostic_procedure
  • chemotherapy                             → Medication
  • radiation                                → Therapeutic_procedure
  • axillary boost                           → Detailed_description
  • tumor                                    → Sign_symptom
  • estrogen receptor - positive             → Detailed_description
  • anti - estrogen therapy                  → Medication
  • tamoxifen                                → Medication
  • 5 years                                  → Duration
  • letrozole                                → Medication
  • 17 years after                           → Duration
  • active                                   → Sign_symptom
  • health                                   → Diagnostic_procedure
  • tachycardia                              → Sign_symptom
  • tachypnea                                → Sign_symptom
  • hypertension                             → Sign_symptom
  • distention                               → Sign_symptom
  • S3                                       → Sign_symptom
  • pulmonary rales                          → Sign_symptom
  • edema                                    → Sign_symptom
  • laboratory values                        → Diagnostic_procedure
  • within normal limits                     → Lab_value
  • N - terminal pro - brain natriuretic peptide → Diagnostic_procedure
  • > 2,000 pg / mL                          → Lab_value
  • cardiomyopathy                           → Disease_disorder
  • cardiac enzyme                           → Diagnostic_procedure
  • electrocardiogram                        → Diagnostic_procedure
  • ECG                                      → Diagnostic_procedure
  • ischemic changes                         → Sign_symptom
  • angiogram                                → Diagnostic_procedure
  • normal                                   → Lab_value
  • ECG                                      → Diagnostic_procedure
  • tachycardia                              → Sign_symptom
  • left - axis deviation                    → Sign_symptom
  • enlargement                              → Sign_symptom
  • QRS complexes                            → Sign_symptom
  • ST changes                               → Sign_symptom
  • echocardiogram                           → Diagnostic_procedure
  • left ventricular ejection fraction       → Diagnostic_procedure
  • LVEF                                     → Diagnostic_procedure
  • 0.20                                     → Lab_value
  • hypokinesis                              → Sign_symptom
  • dilated                                  → Sign_symptom
  • left atrium                              → Biological_structure
  • cardiovascular magnetic resonance        → Diagnostic_procedure
  • CMR                                      → Diagnostic_procedure
  • LVEF                                     → Diagnostic_procedure
  • 0.20                                     → Lab_value
  • T2 - weighted sequence                   → Diagnostic_procedure
  • dysfunction                              → Disease_disorder
  • edema                                    → Sign_symptom
  • gadolinium enhancement                   → Diagnostic_procedure
  • thinning                                 → Sign_symptom
  • treated medically                        → Therapeutic_procedure
  • symptoms                                 → Sign_symptom
  • improved                                 → Lab_value
  • therapy                                  → Therapeutic_procedure
  • β - blocker                              → Medication
  • angiotensin - converting enzyme inhibitor → Medication
  • digoxin                                  → Medication
  • diuretic                                 → Medication
  • therapy                                  → Coreference
  • tapered                                  → Lab_value
  • LVEF                                     → Diagnostic_procedure
  • increased                                → Lab_value
  • 0.20                                     → Lab_value
  • 0.55                                     → Lab_value
  • 8 - month period                         → Duration
  • medications                              → Medication
  • discontinued                             → Lab_value
  • after 1 year                             → Date
  • asymptomatic                             → Sign_symptom

#### Entity Distribution Summary:
               Entity_Type  Count  Percentage               Tier
7     Diagnostic_procedure  11747       20.70   Frequent (>1000)
5             Sign_symptom   6563       11.56   Frequent (>1000)
10               Lab_value   6555       11.55   Frequent (>1000)
11    Biological_structure   5119        9.02   Frequent (>1000)
8     Detailed_description   3279        5.78   Frequent (>1000)
13                    Date   3050        5.37   Frequent (>1000)
3         Disease_disorder   2894        5.10   Frequent (>1000)
2                  History   2440        4.30   Frequent (>1000)
19   Therapeutic_procedure   2086        3.68   Frequent (>1000)
9               Medication   2070        3.65   Frequent (>1000)
0                      Age   1904        3.35   Frequent (>1000)
24                  Dosage   1473        2.60   Frequent (>1000)
6                 Duration   1189        2.09   Frequent (>1000)
4           Clinical_event   1091        1.92   Frequent (>1000)
14  Nonbiological_location   1066        1.88   Frequent (>1000)
29          Family_history    767        1.35  Medium (100-1000)
20             Coreference    484        0.85  Medium (100-1000)
1                      Sex    366        0.64  Medium (100-1000)
12                Distance    338        0.60  Medium (100-1000)
40            Other_entity    323        0.57  Medium (100-1000)
18                    Area    254        0.45  Medium (100-1000)
28                  Volume    222        0.39  Medium (100-1000)
31                    Time    184        0.32  Medium (100-1000)
15               Frequency    180        0.32  Medium (100-1000)
33                Activity    159        0.28  Medium (100-1000)
23             Other_event    141        0.25  Medium (100-1000)
21     Personal_background    130        0.23  Medium (100-1000)
25          Administration    113        0.20  Medium (100-1000)
17                 Subject     92        0.16        Rare (<100)
27              Occupation     82        0.14        Rare (<100)
16                 Outcome     81        0.14        Rare (<100)
32                   Shape     62        0.11        Rare (<100)
26                Severity     46        0.08        Rare (<100)
22     Qualitative_concept     44        0.08        Rare (<100)
34    Quantitative_concept     44        0.08        Rare (<100)
30                 Texture     43        0.08        Rare (<100)
36                   Color     38        0.07        Rare (<100)
35                  Height     16        0.03        Rare (<100)
38                  Weight     16        0.03        Rare (<100)
39    Biological_attribute      6        0.01        Rare (<100)
37                    Mass      4        0.01        Rare (<100)

Tier Breakdown:
Tier
Frequent (>1000)     15
Medium (100-1000)    13
Rare (<100)          13

#### Document-Level Statistics:
  Mean entities per doc: 66.72
  Median entities per doc: 66.00
  Std dev: 36.76
  Min: 4, Max: 187

#### Entity Type Coverage (% of docs containing type):
               Entity_Type  Coverage_%
10                     Age       98.00
0     Diagnostic_procedure       95.50
1             Sign_symptom       94.50
17                     Sex       91.50
6         Disease_disorder       90.00
3     Biological_structure       84.50
2                Lab_value       83.25
13          Clinical_event       81.75
5                     Date       77.00
4     Detailed_description       74.50
8    Therapeutic_procedure       73.00
9               Medication       72.00
14  Nonbiological_location       60.25
7                  History       57.00
12                Duration       55.00
16             Coreference       42.00
11                  Dosage       35.00
26     Personal_background       22.50
18                Distance       21.00
15          Family_history       16.75
30                 Outcome       16.75
24                Activity       14.25
27          Administration       14.00
23               Frequency       12.00
20                    Area       10.50
28                 Subject       10.00
32                Severity        9.50
21                  Volume        7.50
22                    Time        7.25
29              Occupation        6.50
31                   Shape        4.50
25             Other_event        4.25
35                 Texture        3.75
36                   Color        3.00
34    Quantitative_concept        2.75
33     Qualitative_concept        2.50
19            Other_entity        2.25
37                  Height        2.00
38                  Weight        2.00
39    Biological_attribute        0.50
40                    Mass        0.50

#### Rare Entity Examples: 
Weight: 8 total, 4 unique
  Examples: ['3132 g', '72 kg', '47 kg', '83 kg']

Texture: 15 total, 8 unique
  Examples: ['severely tender', 'cobble stone', 'smooth surface', 'classical " sandstorm " appearance', 'surrounding ground glass opacity']

Color: 16 total, 8 unique
  Examples: ['pale tan', 'homogenously white', 'coffee - colored', 'grayish - white', 'gray - white']

Quantitative_concept: 23 total, 16 unique
  Examples: ['42', '6/10', 'six times', '10–15 cigarettes per day', '20']

Occupation: 26 total, 13 unique
  Examples: ['student', 'college student', 'company worker', 'a dog breeder for 12 years', 'cotton farmer']

Qualitative_concept: 26 total, 12 unique
  Examples: ['high', 'improved', 'daytime only', 'normal', 'multiple']

Shape: 30 total, 13 unique
  Examples: ['nodular appearance', 'predominantly pseudoalveolar pattern', 'spindle - shaped', 'round', 'irregular edge']

Severity: 40 total, 12 unique
  Examples: ['suddenly worsened', 'extensive', 'rapidly', 'worsened', 'benign']

Subject: 66 total, 25 unique
  Examples: ['mother', 'Parents', 'ENT specialist', 'daughter', 'sibling']

Outcome: 69 total, 14 unique
  Examples: ['returned', 'died', 'doing well', 'recovered', 'asymptomatic']

## Models

### PubMedBERT Model
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" # PubMedBERT
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2tag,
    label2id=tag2id
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### Bioformer-L Model

### BioBERT Model