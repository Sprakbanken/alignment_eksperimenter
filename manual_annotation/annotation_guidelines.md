# Annotation guidelines
The motivation for annotating this data is to have a good evaluation set for measuring precision and recall of the bitext mining pipeline.
We want to have high quality examples of parallell document pairs, and not-parallel document pairs.
Therefore, all texts with bad quality should be filtered away in this phase (see Faulty below.)

Because we care about the meaning of the documents, we allow some variation in sentence order, as long as all the important information is present in both examples in the document pair.


## Annotation categories
Each document pair must be annotated with one of the following marks:

### Parallel
All sentences must be present in both documents for them to be considered parallel.  
The sentences must be mostly in the same order.
 
### Almost parallel 
Minor details differ between the pairs, for instance: 
- a few sentences missing/added (should be few sentences relative to document length, use your judgement)
- sentence order is jumbled, but mostly contain the same information 
- a few numbers or dates differ, but the rest of the text is the same 

### Not parallel
The documents in the pair do not share enough similarities to be considered parallel or almost parallel
 
### Faulty
This pair should be filtered out from the dataset due to insufficient text quality.

Examples include: 
- The language annotation of one or more documents in the pair in incorrect
- The document does not contain natural language 

## Additional annotaion
We also flag document pairs that contain one or more of the following personal information (to be cleaned before publication):
- Full name
- Private telephone number (i.e. not customer service etc)
- Private email (i.e. not customer service etc)
- Private address (i.e. not office etc)
