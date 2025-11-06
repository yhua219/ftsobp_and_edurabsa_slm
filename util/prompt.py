"""
##########################################################
NOTICE
Code in this script is licensed and distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
You must provide attribution. You may not use the code for commercial purposes. Any derivatives must be shared under the same license.

Copyright (c) 2025 Authors of [Data-Efficient Adaptation and a Novel Evaluation Method for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2511.03034).
##########################################################
"""


all_prompt_dict = {

    'V1_zeroshot':  # zeroshot, output as a whole string (instead of list of strs). Used for both train/val and dev/test.

{
    'OE':
"""### Instruction: 
Given the input text, extract ALL opinion expressions about the course, staff, or university. 
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).

**Rules:**
- Extract each opinion expression VERBATIM and as CONSECUTIVE tokens.
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Use these specific tags for each extracted opinion expression: <opn>opinion expressions</opn>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Tag-wrapped units MUST be separated by commas

**Output format:** 
[<opn>...</opn>, <opn>...</opn>, ..., <opn>...</opn>]
""",


    'AOPE':
"""### Instruction:
Given the input text, extract ALL pairs of opinion expressions and their corresponding aspect terms about the course, staff, or university.
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).
Aspect terms are opinion targets. Only use a pronoun if you cannot find a direct aspect term in the same sentence or adjacent context. 
Each aspect-opinion combination is a pair. 

**Rules:**
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Extract all opinion and aspect terms VERBATIM and as CONSECUTIVE tokens. 
- Use 'null' for implicit aspects. Opinions cannot be null.
- If an aspect is mapped to multiple opinion expressions, or vice versa, extract each 1:1 pair separately. 
- Use these specific tags for each component within each pair: <asp>aspect terms</asp>, <opn>opinion expressions</opn>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Pairs MUST be separated by commas

**Output format:** 
[<asp>...</asp><opn>...</opn>, <asp>...</asp><opn>...</opn>, ..., <asp>...</asp><opn>...</opn>]
""",


    'AOC':
"""### Instruction:
Given the input text, extract ALL pairs of opinion expressions and their corresponding aspect terms about the course, staff, or university. Then classify the category for each aspect-opinion pair.
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).
Aspect terms are opinion targets. Only use a pronoun if you cannot find a direct aspect term in the same sentence or adjacent context. 
Each aspect-opinion-category combination is a triplet. 

**Rules:**
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Extract all opinion and aspect terms VERBATIM and as CONSECUTIVE tokens. 
- Use 'null' for implicit aspects. Opinions cannot be null.
- If an aspect is mapped to multiple opinion expressions, or vice versa, extract each 1:1 pair separately. 
- Categorise each aspect-opinion pair first into one main category (the keys) in the category_mapping below, and then into one of its appropriate subcategories (values for the key). The category label follows "Main category - subcategory" format.
category_mapping = {
  "Course": ["Content", "Learning activity", "Assessment", "Workload", "Difficulty", "Course materials", "Technology & tools", "Overall"],
  "Staff": ["Teaching", "Knowledge & skills", "Helpfulness", "Attitude", "Personal traits", "Overall"],
  "University": ["Cost", "Opportunities", "Programme", "Campus & facilities", "Culture & diversity", "Information & Services", "Social engagement & activities", "Overall"]
}

- Use these specific tags for each component within each triplet: <asp>aspect terms</asp>, <opn>opinion expressions</opn>, <cat>category</cat>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Triplets MUST be separated by commas

**Output format:** 
[<asp>...</asp><opn>...</opn><cat>...</cat>, <asp>...</asp><opn>...</opn><cat>...</cat>, ..., <asp>...</asp><opn>...</opn><cat>...</cat>]
""",


    'ASTE':
"""### Instruction:
Given the input text, extract ALL pairs of opinion expressions and their corresponding aspect terms about the course, staff, or university. Then classify the sentiment for each aspect-opinion pair.
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).
Aspect terms are opinion targets. Only use a pronoun if you cannot find a direct aspect term in the same sentence or adjacent context. 
Each aspect-opinion-sentiment combination is a triplet. 

**Rules:**
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Extract all opinion and aspect terms VERBATIM and as CONSECUTIVE tokens. 
- Use 'null' for implicit aspects. Opinions cannot be null.
- If an aspect is mapped to multiple opinion expressions, or vice versa, extract each 1:1 pair separately. 
- Classify the sentiment into one of 'positive', 'neutral', 'negative'. 
- Use these specific tags for each component within each triplet: <asp>aspect terms</asp>, <opn>opinion expressions</opn>, <sen>sentiment</sen>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Triplets MUST be separated by commas

**Output format:** 
[<asp>...</asp><opn>...</opn><sen>...</sen>, <asp>...</asp><opn>...</opn><sen>...</sen>, ..., <asp>...</asp><opn>...</opn><sen>...</sen>]
""",


    'ASQE':
"""### Instruction:
Given the input text, extract ALL pairs of opinion expressions and their corresponding aspect terms about the course, staff, or university. Then classify the category and sentiment for each aspect-opinion pair.
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).
Aspect terms are opinion targets. Only use a pronoun if you cannot find a direct aspect term in the same sentence or adjacent context. 
Each aspect-opinion-category-sentiment combination is a quadruplet. 

**Rules:**
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Extract all opinion and aspect terms VERBATIM and as CONSECUTIVE tokens. 
- Use 'null' for implicit aspects. Opinions cannot be null.
- If an aspect is mapped to multiple opinion expressions, or vice versa, extract each 1:1 pair separately. 
- Categorise each aspect-opinion pair first into one main category (the keys) in the category_mapping below, and then into one of its appropriate subcategories (values for the key). The category label follows "Main category - subcategory" format.
category_mapping = {
  "Course": ["Content", "Learning activity", "Assessment", "Workload", "Difficulty", "Course materials", "Technology & tools", "Overall"],
  "Staff": ["Teaching", "Knowledge & skills", "Helpfulness", "Attitude", "Personal traits", "Overall"],
  "University": ["Cost", "Opportunities", "Programme", "Campus & facilities", "Culture & diversity", "Information & Services", "Social engagement & activities", "Overall"]
}

- Classify the sentiment into one of 'positive', 'neutral', 'negative'. 
- Use these specific tags for each component within each quadruplet: <asp>aspect terms</asp>, <opn>opinion expressions</opn>, <cat>category</cat>, <sen>sentiment</sen>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Quadruplets MUST be separated by commas

**Output format:** 
[<asp>...</asp><opn>...</opn><cat>...</cat><sen>...</sen>, <asp>...</asp><opn>...</opn><cat>...</cat><sen>...</sen>, ..., <asp>...</asp><opn>...</opn><cat>...</cat><sen>...</sen>]
"""

},


#===============================================
#===============================================

    'V1_fewshot':

{
    'OE':
"""### Instruction: 
Given the input text, extract ALL opinion expressions about the course, staff, or university. 
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).

**Rules:**
- Extract each opinion expression VERBATIM and as CONSECUTIVE tokens.
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Use these specific tags for each extracted opinion expression: <opn>opinion expressions</opn>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Tag-wrapped units MUST be separated by commas

**Output format:** 
[<opn>...</opn>, <opn>...</opn>, ..., <opn>...</opn>]

### Examples:
Input: "The professor was knowledgeable but the assignments were too hard."
Output: [<opn>knowledgeable</opn>, <opn>too hard</opn>]

Input: "I strongly recommend it."
Output: [<opn>strongly recommend</opn>]

Input: "She never reply to emails or answer questions"
Output: [<opn>never reply to emails or answer questions</opn>]

Input: "There were 10 assignments, 5 quizzes, 1 final exam."
Output: [<opn></opn>]
""",


    'AOPE':
"""### Instruction:
Given the input text, extract ALL pairs of opinion expressions and their corresponding aspect terms about the course, staff, or university.
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).
Aspect terms are opinion targets. Only use a pronoun if you cannot find a direct aspect term in the same sentence or adjacent context. 
Each aspect-opinion combination is a pair. 

**Rules:**
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Extract all opinion and aspect terms VERBATIM and as CONSECUTIVE tokens. 
- Use 'null' for implicit aspects. Opinions cannot be null.
- If an aspect is mapped to multiple opinion expressions, or vice versa, extract each 1:1 pair separately. 
- Use these specific tags for each component within each pair: <asp>aspect terms</asp>, <opn>opinion expressions</opn>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Pairs MUST be separated by commas

**Output format:** 
[<asp>...</asp><opn>...</opn>, <asp>...</asp><opn>...</opn>, ..., <asp>...</asp><opn>...</opn>]

### Examples:
Input: "The professor was knowledgeable but the assignments were too hard."
Output: [<asp>professor</asp><opn>knowledgeable</opn>, <asp>assignments</asp><opn>too hard</opn>]

Input: "I strongly recommend it."
Output: [<asp>null</asp><opn>strongly recommend</opn>]

Input: "She never reply to emails or answer questions"
Output: [<asp>She</asp><opn>never reply to emails or answer questions</opn>]

Input: "There were 10 assignments, 5 quizzes, 1 final exam."
Output: [<asp></asp><opn></opn>]
""",


    'AOC':
"""### Instruction:
Given the input text, extract ALL pairs of opinion expressions and their corresponding aspect terms about the course, staff, or university. Then classify the category for each aspect-opinion pair.
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).
Aspect terms are opinion targets. Only use a pronoun if you cannot find a direct aspect term in the same sentence or adjacent context. 
Each aspect-opinion-category combination is a triplet. 

**Rules:**
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Extract all opinion and aspect terms VERBATIM and as CONSECUTIVE tokens. 
- Use 'null' for implicit aspects. Opinions cannot be null.
- If an aspect is mapped to multiple opinion expressions, or vice versa, extract each 1:1 pair separately. 
- Categorise each aspect-opinion pair first into one main category (the keys) in the category_mapping below, and then into one of its appropriate subcategories (values for the key). The category label follows "Main category - subcategory" format.
category_mapping = {
  "Course": ["Content", "Learning activity", "Assessment", "Workload", "Difficulty", "Course materials", "Technology & tools", "Overall"],
  "Staff": ["Teaching", "Knowledge & skills", "Helpfulness", "Attitude", "Personal traits", "Overall"],
  "University": ["Cost", "Opportunities", "Programme", "Campus & facilities", "Culture & diversity", "Information & Services", "Social engagement & activities", "Overall"]
}

- Use these specific tags for each component within each triplet: <asp>aspect terms</asp>, <opn>opinion expressions</opn>, <cat>category</cat>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Triplets MUST be separated by commas

**Output format:** 
[<asp>...</asp><opn>...</opn><cat>...</cat>, <asp>...</asp><opn>...</opn><cat>...</cat>, ..., <asp>...</asp><opn>...</opn><cat>...</cat>]

### Examples:
Input: "The professor was knowledgeable but the assignments were too hard."
Output: [<asp>professor</asp><opn>knowledgeable</opn><cat>Staff - Knowledge & skills</cat>, <asp>assignments</asp><opn>too hard</opn><cat>Course - Assessment</cat>]

Input: "It was disappointing overall."
Output: [<asp>null</asp><opn>disappointing</opn><cat>Course - Overall</cat>]

Input: "She never reply to emails or answer questions"
Output: [<asp>She</asp><opn>never reply to emails or answer questions</opn><cat>Staff - Helpfulness</cat>]

Input: "There were 10 assignments, 5 quizzes, 1 final exam."
Output: [<asp></asp><opn></opn><cat></cat>]
""",


    'ASTE':
"""### Instruction:
Given the input text, extract ALL pairs of opinion expressions and their corresponding aspect terms about the course, staff, or university. Then classify the sentiment for each aspect-opinion pair.
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).
Aspect terms are opinion targets. Only use a pronoun if you cannot find a direct aspect term in the same sentence or adjacent context. 
Each aspect-opinion-sentiment combination is a triplet. 

**Rules:**
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Extract all opinion and aspect terms VERBATIM and as CONSECUTIVE tokens. 
- Use 'null' for implicit aspects. Opinions cannot be null.
- If an aspect is mapped to multiple opinion expressions, or vice versa, extract each 1:1 pair separately. 
- Classify the sentiment into one of 'positive', 'neutral', 'negative'. 
- Use these specific tags for each component within each triplet: <asp>aspect terms</asp>, <opn>opinion expressions</opn>, <sen>sentiment</sen>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Triplets MUST be separated by commas

**Output format:** 
[<asp>...</asp><opn>...</opn><sen>...</sen>, <asp>...</asp><opn>...</opn><sen>...</sen>, ..., <asp>...</asp><opn>...</opn><sen>...</sen>]

### Examples:
Input: "The professor was knowledgeable but the assignments were too hard."
Output: [<asp>professor</asp><opn>knowledgeable</opn><sen>positive</sen>, <asp>assignments</asp><opn>too hard</opn><sen>negative</sen>]

Input: "It was disappointing overall."
Output: [<asp>null</asp><opn>disappointing</opn><sen>negative</sen>]

Input: "She never reply to emails or answer questions"
Output: [<asp>She</asp><opn>never reply to emails or answer questions</opn><sen>negative</sen>]

Input: "There were 10 assignments, 5 quizzes, 1 final exam."
Output: [<asp></asp><opn></opn><sen></sen>]
""",


    'ASQE':
"""### Instruction:
Given the input text, extract ALL pairs of opinion expressions and their corresponding aspect terms about the course, staff, or university. Then classify the category and sentiment for each aspect-opinion pair.
Opinion expressions are words/phrases expressing evaluation, feeling, or judgment (including both explicit and implicit opinions, not objective facts).
Aspect terms are opinion targets. Only use a pronoun if you cannot find a direct aspect term in the same sentence or adjacent context. 
Each aspect-opinion-category-sentiment combination is a quadruplet. 

**Rules:**
- Extract EVERY opinion in the text, including both explicit and implicit opinion expressions.
- Extract all opinion and aspect terms VERBATIM and as CONSECUTIVE tokens. 
- Use 'null' for implicit aspects. Opinions cannot be null.
- If an aspect is mapped to multiple opinion expressions, or vice versa, extract each 1:1 pair separately. 
- Categorise each aspect-opinion pair first into one main category (the keys) in the category_mapping below, and then into one of its appropriate subcategories (values for the key). The category label follows "Main category - subcategory" format.
category_mapping = {
  "Course": ["Content", "Learning activity", "Assessment", "Workload", "Difficulty", "Course materials", "Technology & tools", "Overall"],
  "Staff": ["Teaching", "Knowledge & skills", "Helpfulness", "Attitude", "Personal traits", "Overall"],
  "University": ["Cost", "Opportunities", "Programme", "Campus & facilities", "Culture & diversity", "Information & Services", "Social engagement & activities", "Overall"]
}

- Classify the sentiment into one of 'positive', 'neutral', 'negative'. 
- Use these specific tags for each component within each quadruplet: <asp>aspect terms</asp>, <opn>opinion expressions</opn>, <cat>category</cat>, <sen>sentiment</sen>

**Critical formatting requirements:**
- Output MUST be a valid Python list
- Quadruplets MUST be separated by commas

**Output format:** 
[<asp>...</asp><opn>...</opn><cat>...</cat><sen>...</sen>, <asp>...</asp><opn>...</opn><cat>...</cat><sen>...</sen>, ..., <asp>...</asp><opn>...</opn><cat>...</cat><sen>...</sen>]

### Examples:
Input: "The professor was knowledgeable but the assignments were too hard."
Output: [<asp>professor</asp><opn>knowledgeable</opn><cat>Staff - Knowledge & skills</cat><sen>positive</sen>, <asp>assignments</asp><opn>too hard</opn><cat>Course - Assessment</cat><sen>negative</sen>]

Input: "It was disappointing overall."
Output: [<asp>null</asp><opn>disappointing</opn><cat>Course - Overall</cat><sen>negative</sen>]

Input: "She never reply to emails or answer questions"
Output: [<asp>She</asp><opn>never reply to emails or answer questions</opn><cat>Staff - Helpfulness</cat><sen>negative</sen>]

Input: "There were 10 assignments, 5 quizzes, 1 final exam."
Output: [<asp></asp><opn></opn><cat></cat><sen></sen>]
"""

},



}
