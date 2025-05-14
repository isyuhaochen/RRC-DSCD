get_statement = '''
From the provided article, extract exactly five sentences that correspond to one of the following categories: Attitudinal, Definition, Logical, Factual, Scope, or Temporal.
Format your response as a Python-parsable list with exactly five elements, like this:
["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"].
Article:
'''

test_example = '''
"Will you be my valentine?" I prayed to hear those three magical letters, and my prayer was heard."Yes, of course!" Her heart was mine. The next day, February 14th, I spent all day dreaming of her. Her deep blue eyes, her long, black hair, but most importantly, her heart. Her heart was mine. And then I noticed the time. She would be here in half an hour, and I hadn’t even begun to prepare a meal! The pantry door flew open for me to search for something suitable for the occasion. Cauliflower? Mashed potatoes? Pasta with white sauce? That was my best bet. As I finished plating the meal, the daydreams returned. She was the most beautiful woman I had ever met. And she was mine. Her heart was mine. I placed a white rose against the end of the plate. Just as I had done this, there was a knock at the door. My hand rested against the knob, ready to open the door, before feeling a moment of sorrow. Once the moment passed, I turned the knob and welcomed her inside. She wore a beautiful white dress. It was perfect. I guided her to the dining room, and she reacted just as every valentine in the past. Her face was one of confusion.
"There’s only one plate," she said, puzzled. Just like every valentine in the past. I shushed her. Quietly, I pulled out the knife that had been hidden. I turned to face her and thrust the knife into her chest, careful not to penetrate her heart. She was dead before she could react. Her heart was then carved out. Her blood stained her dress red. Red, the color of love. Her heart was carefully set on the plate. It colored the sauce red. Red, the color of love. And finally, it stained the rose red. Red, the color of love. I feasted the same as I had every year prior, savoring every bite. Her heart was mine, after all. Then, it was time to rest again, and in 11 months, I would find a new love. A new heart to call mine.
'''

content_insert = '''
Article:{Article}
Your task is to generate a contradicted_sentence that contradicts the original sentence: {Statement} in one of the following ways: Attitudinal, Definition, Logical, Factual, Scope, or Temporal. The contradictory sentence should be inserted at the appropriate position within the given article.  
If there are other_contradictory_sentences, it must have a clear contradiction with contradicted_sentence.

Attitudinal: The contradiction arises from a difference in opinion or feeling.
Definition: The contradiction arises from a difference in the meaning of a word or concept.
Logical: The contradiction arises from a logical inconsistency.
Factual: The contradiction arises from a difference in facts or events.
Scope: The contradiction arises from a difference in the scope of a statement.
Temporal: The contradiction arises from a difference in time or sequence of events.

These contradictions must indicate a scenario which cannot coexist with the original statements. Remember, the goal is not to negate the original statement but to offer a fact that renders the original fact impossible. And you need to insert the contradicted_sentence into the appropriate position.

Good Examples:
original_sentence: Tim Johnson, ... He was born in the UK and moved to Mexico with his family in 2006.
contradicted_sentence: Tim Johnson was born in Paris.
other_contradictory_sentences: ['Tim Johnson is British because he was born in the UK']

original_sentence: Tom's wife, Lisa, is an accountant and she earns much more than Tom.
contradicted_sentence: Lisa is Tom's sister.
other_contradictory_sentences: ['Tom and Lisa got married in 2015']

Bad Example:
original_sentence: Lisa is a professor.
contradicted_sentence: The university's headmaster is Lisa.
(This is a poor example since one can simultaneously be a professor and a headmaster, therefore, these are not mutually exclusive scenarios.)
other_contradictory_sentences: ['Today is a good day']
(Unrelated to the original sentence and the modified sentence)

You need to return the result in the following JSON format, ensuring it is fully parsable by Python:  
{{
  "original_sentence": "xxx",  // The original sentence
  "contradicted_sentence": "xxx",  // The newly generated contradictory sentence
  "insert_position_sentence": "xxx",  // The sentence immediately before the insertion point
  "next_sentence_after_insert": "xxx",  // The sentence immediately after the insertion point
  "other_contradictory_sentences": ["sentence1", "sentence2", ...]  // Other sentences in the text that also become contradictory due to the inserted sentence; return an empty list [] if none exist
  "contradiction_type": "xxx",  // The type of contradiction: Attitudinal, Definition, Logical, Factual, Scope, or Temporal
  "contradiction_reason": "xxx" // The reason for the contradiction.
}}
Ensure that the contradiction is meaningful and aligns with one of the specified contradiction types. Please ensure that the added sentence (contradicted_sentence) has a clear conflict with the original sentence. If no such conflicts arise, return `none`.
'''

content_replace = '''
Article:{Article}
Your task is to modify the sentences: {Statement} in the article. The modifications you make should introduce a contradiction to other sentence in one of the following areas: Attitudinal, Definition, Logical, Factual, Scope, or Temporal. 
If there are other_contradictory_sentences, it must have a clear contradiction with modified_sentence.

Attitudinal: The contradiction arises from a difference in opinion or feeling.
Definition: The contradiction arises from a difference in the meaning of a word or concept.
Logical: The contradiction arises from a logical inconsistency.
Factual: The contradiction arises from a difference in facts or events.
Scope: The contradiction arises from a difference in the scope of a statement.
Temporal: The contradiction arises from a difference in time or sequence of events.

These contradictions must indicate a scenario which cannot coexist with the original statements. Remember, the goal is not to negate the original statement but to offer a fact that renders the original fact impossible. 

Good Examples:
original_sentence: Tim Johnson, ... He was born in the UK and moved to Mexico with his family in 2006.
modified_sentence: Tim Johnson was born in Paris.
other_contradictory_sentences: ['Tim Johnson is British because he was born in the UK']

original_sentence: Tom's wife, Lisa, is an accountant and she earns much more than Tom.
modified_sentence: Lisa is Tom's sister.
other_contradictory_sentences: ['Tom and Lisa got married in 2015']

Bad Example:
original_sentence: Lisa is a professor.
modified_sentence: The university's headmaster is Lisa.
(This is a poor example since one can simultaneously be a professor and a headmaster, therefore, these are not mutually exclusive scenarios.)
other_contradictory_sentences: ['Today is a good day','Lisa is a old professor.']
(Unrelated and Uncontradicted to the the modified sentence and other_contradictory_sentences)

You need to return the result in the following JSON format, ensuring it is fully parsable by Python:  
{{
    "original_sentence": "xxx",  // The original sentence
    "modified_sentence": "xxx",  // The modified sentence that introduces a contradiction, return empty string "", if can't modified sentence to introduce contradiction
    "other_contradictory_sentences": ["sentence1", "sentence2", ...],  // other_contradictory_sentences in the article (do not contain original_sentence and modified_sentence) contradiction with modified_sentence; return an empty list [] if none exist
    "contradiction_type": "xxx",  // The type of contradiction: Attitudinal, Definition, Logical, Factual, Scope, or Temporal
    "contradiction_reason": "xxx" // The reason for the contradiction.
}}
Ensure that the contradiction is meaningful and aligns with one of the specified contradiction types. Please ensure that the modified_sentence has a clear conflict with other_contradictory_sentences. If no such conflicts arise, return `none`.
'''

content_swap = ''' 
Article: {Article}
Your task is to swap certain sentences in the article to introduce a contradiction (Changing the order of some sentences may introduce some contradiction, return an empty list [] in "modified_sentence_order" if can't introduce contradiction). The sentences you swap should create a contradiction in one of the following areas: Attitudinal, Definition, Logical, Factual, Scope, or Temporal.
If there are other_contradictory_sentences, it must have a clear contradiction with modified_sentence_order.

Attitudinal: The contradiction arises from a difference in opinion or feeling.
Definition: The contradiction arises from a difference in the meaning of a word or concept.
Logical: The contradiction arises from a logical inconsistency.
Factual: The contradiction arises from a difference in facts or events.
Scope: The contradiction arises from a difference in the scope of a statement.
Temporal: The contradiction arises from a difference in time or sequence of events.

Good Examples:
original_sentence_order: ['The industrial revolution led to rapid urbanization.','Many people moved to cities in search of jobs.','Overcrowding and poor living conditions became major issues.']
modified_sentence_order: ['Overcrowding and poor living conditions became major issues.','Many people moved to cities in search of jobs.','The industrial revolution led to rapid urbanization.']
other_contradictory_sentences: ['After the Industrial Revolution, people moved towards cities']
(After this adjustment, crowded and poor living conditions emerged first, and then people migrated in large numbers, while the Industrial Revolution occurred last, which goes against historical facts and causal relationships.)

Bad Example:
original_sentence_order: ['She woke up early in the morning.','She made herself a cup of coffee.','She read the newspaper before leaving for work.']
modified_sentence_order: ['She read the newspaper before leaving for work.','She made herself a cup of coffee.','She woke up early in the morning.']
(After changing the order, Attitude was not introduced, Definition, Logical, Factual, Scope, or Temporary conflict, only the narrative order has changed.)
other_contradictory_sentences: ['Today is a good day','Lisa is a old professor.']
(The order of these events can be changed without affecting the overall rationality)

Return the result in the following JSON format, ensuring it is fully parsable by Python:
{{
    "original_sentence_order": ["sentence1", "sentence2", "sentence3", ...],  // The original order of sentences that were swapped (only include swapped sentences).
    "modified_sentence_order": ["sentence3", "sentence1", "sentence2", ...],  // The new order of the swapped sentences (you may make slight changes to wording if necessary). If no sentences are swapped to create a contradiction, return an empty list [].
    "other_contradictory_sentences": ["sentence4", "sentence5", ...],  // Other sentences in the article that also become contradictory due to the swap (do not include the sentences in "original_sentence_order" or "modified_sentence_order"). Return an empty list [] if none exist.
    "contradiction_type": "xxx",  // The type of contradiction: Attitudinal, Definition, Logical, Factual, Scope, or Temporal.
    "contradiction_reason": "xxx" // The reason for the contradiction.
}}
Ensure that the contradiction you introduce is meaningful and aligns with one of the specified types. Please ensure that altering the sentence order creates clear and undeniable contradictions within the article. If no such conflicts arise, return `none`.
'''

content_delete = '''
Article: {Article}  
Your task is to identify three sentences—A, B, and C—in the article(SentenceA and SentenceC should be conflicting, but the article contains SentenceB, which leads to A and B not conflicting, Sentence C and sentence A cannot hold at the same time). Removing sentence B should create a meaningful contradiction in one of the following categories: Attitudinal, Definition, Logical, Factual, Scope, or Temporal. Sentences A and C should be logically linked in a way that removing B causes a contradiction between them.  
If there are other_contradictory_sentences, it must have a clear contradiction with sentencesA and sentencesC.

Attitudinal: The contradiction arises from a difference in opinion or feeling.
Definition: The contradiction arises from a difference in the meaning of a word or concept.
Logical: The contradiction arises from a logical inconsistency.
Factual: The contradiction arises from a difference in facts or events.
Scope: The contradiction arises from a difference in the scope of a statement.
Temporal: The contradiction arises from a difference in time or sequence of events.

Good Examples:
sentencesA: The ancient manuscript was declared a forgery by experts.
sentencesB: However, subsequent analysis revealed traces of ink consistent with the historical period.
sentencesC: Historians now consider it one of the most important discoveries of the century.
(If Sentence B is removed, it becomes contradictory how a document initially deemed a forgery is later considered a major discovery.)
other_contradictory_sentences:['Due to its importance, it is now housed in the National Museum']

sentencesA: The project was expected to be completed within a year.
sentencesB: However, unexpected delays in the supply chain caused significant setbacks.
sentencesC: As a result, the deadline was extended by six months.
(If Sentence B is removed, the shift in the timeline becomes unclear, creating a contradiction between the initial deadline and the final extension.)
other_contradictory_sentences:['']

Bad Example:
sentencesA: The meeting was scheduled for 3 PM.
sentencesB: Due to a scheduling error, it was moved to 4 PM.
sentencesC: Everyone arrived on time for the meeting.
(If Sentence B is removed, there is no conflict. Sentence A simply tells the scheduled time, and Sentence C remains unaffected, as it only mentions the punctuality of attendees.)
other_contradictory_sentences:['today is a good day']
(Unrelated and Uncontradicted to the sentencesA and sentencesC)

sentencesA: The era of global supply chain dominance is being incrementally rivaled by the resurgence of localized supply networks.
sentencesB: SMEs, traditionally reliant on larger supply chains, have been adapting to this shift in dynamic.
sentencesC: With rising transportation costs and the need for faster lead times, there has been an upward trend in 'proximity manufacturing.'
(If Sentence B is removed, there is no conflict. Sentence A and Sentence C remain unaffected, as they are independent statements about the global supply chain and localized supply networks.)

Return the result in the following JSON format, ensuring it is fully parsable by Python:  
{{
    "sentencesA": "xxx",  // Sentence A, which will conflict with Sentence C if Sentence B is removed.
    "sentencesB": "xxx",  // The key sentence whose removal introduces the contradiction. return empty string "" if can't find the sentence to remove.
    "sentencesC": "xxx",  // Sentence C, which will conflict with Sentence A if Sentence B is removed.
    "other_contradictory_sentences": ["sentence1", "sentence2", ...],  // Any other sentences in the article that also become contradictory due to the deletion of Sentence B. Return an empty list [] if none exist.
    "contradiction_type": "xxx", // The type of contradiction: Attitudinal, Definition, Logical, Factual, Scope, or Temporal.
    "contradiction_reason": "xxx" // The reason for the contradiction.
}}  
Ensure that the contradiction you introduce is meaningful and aligns with one of the specified types. Please ensure that after deleting sentence B, there is a clear conflict between sentences A and C (Sentence C and sentence A cannot hold at the same time). If no such conflicts arise, return `none`. 
'''

