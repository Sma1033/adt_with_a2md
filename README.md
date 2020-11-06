### Supplementary material for paper:
“IMPROVING AUTOMATIC DRUM TRANSCRIPTION USING LARGE-SCALE AUDIO-TO-MIDI ALIGNED DATA”

----------------

###  ***  IMPORTANT NOTICE !!  *** <br />
The numbers reported in the third column of Table 2 (B-CNN, 3-fold) in our paper are wrong. They should be 0.75, 0.72, and 0.60 for the three public datasets (marked red below)

<img src="https://raw.githubusercontent.com/Sma1033/adt_with_a2md/main/pics/new_table2.png" style="zoom:70%" />

The error is due to the misunderstanding of training material employed in the three-fold validation process. In the previous work [5], there are two different three-fold validation strategy: <br />
`[i]   3-fold, both train/test process are performed on each single dataset ` <br />
`[ii]  3-fold, train on a combination of three public datasets (ENST + MDB-Drums + RBMA13) ` <br />
In our work, we accidentally report B-CNN (3-fold) evaluation result under scenario [i] and compare it with the CNN/CRNN evaluation results using [ii]. The unfair comparison between models may give readers misleading ideas when analyzing the difference between models. To fix this, we re-run a three-fold validation test on our B-CNN (3-fold) model using [ii] and report the latest result in the new Table 2 presented above. In our final version paper, we will update the numbers accordingly. <br />

----------------

### Accompanying materials: <br />

- penalty search: A demo ipynb script to obtain a penalty value (as similarity measurement) for a given audio pair. Please check `penalty_search.ipynb` for more details.

- A2MD dataset: Youtube downloaded tracks and their aligned MIDI files. The files are categorized into seven groups based on the retrieved penalty values. download link is provided here: <br />https://drive.google.com/uc?export=download&id=1ZRbLz7aaJibd9F121LLn_yKPmgtYUqHu <br />