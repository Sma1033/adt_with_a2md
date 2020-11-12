### Supplementary material for paper:
“IMPROVING AUTOMATIC DRUM TRANSCRIPTION USING LARGE-SCALE AUDIO-TO-MIDI ALIGNED DATA”

----------------

###  ***  IMPORTANT CORRECTION !!  *** <br />
The numbers reported in the third column of Table 2 (B-CNN, 3-fold) are presented improperly. They should be 0.75, 0.72, and 0.60 (marked in red below) for the three public datasets.
The results in Table 2 still verify the effectiveness of the proposed model despite the correction.

<img src="https://raw.githubusercontent.com/Sma1033/adt_with_a2md/main/pics/new_table2.png" style="zoom:70%" />

This error was due to the inconsistent configuration of the 3-fold cross validation process. In the previous work [1], the evaluation results based on two different 3-fold strategies were reported on their complementary website (http://ifs.tuwien.ac.at/~vogl/dafx2018/): <br />
`[i] sepearate 3-fold -- 3-fold cross validation on each of the public datasets (ENST + MDB-Drums + RBMA13) ` <br />
`[ii]joint 3-fold -- combine all public datasets and perform 3-fold cross validation jointly. ` <br />
In our submitted manuscript, we included the results of CNN/CRNN (3-fold) [1] under scenario [ii] in our Table 2 for comparison. However, we incorrectly reported the results of B-CNN (3-fold) under scenario [i], which was inconsistent with [1]. To avoid confusions and ensure the compatibility of the results, we have updated the table (as shown above) and will correct the numbers in the manuscript accordingly. <br />

<br />
[1] Richard Vogl, Gerhard Widmer, and Peter Knees,  “To-wards multi-instrument drum transcription,” in Proceedings of International Conference on Digital Audio Ef-fects (DAFx), 2018.

----------------

### Accompanying materials: <br />

- Penalty search: A Python script to demo the process of penalty value (as similarity measurement) retrieval for a given audio pair. Please check file `penalty_search.ipynb` for more details.

- A2MD dataset: 1565 Youtube downloaded tracks and their aligned MIDI files are provided in a single zip file. The audio/MIDI files are categorized into seven groups based on the retrieved penalty values. A2MD download link: <br />https://drive.google.com/uc?export=download&id=1ZRbLz7aaJibd9F121LLn_yKPmgtYUqHu <br />

- Song demo: We provide three demo songs for model comparison. In the demo songs, the original drum track is replaced with re-syntheszed drum track using transcription results from different models. It should be noted that only three basic drum instruments (i.e., kick drum, snare drum, hihat) are used in the the re-synthesized track. Please check directory `demo_songs` for audio tracks.

<img src="https://raw.githubusercontent.com/Sma1033/adt_with_a2md/main/demo_songs/song01_tenderness/tracks.png" style="zoom:70%" />
<img src="https://raw.githubusercontent.com/Sma1033/adt_with_a2md/main/demo_songs/song02_billie_jean/tracks.png" style="zoom:70%" />
<img src="https://raw.githubusercontent.com/Sma1033/adt_with_a2md/main/demo_songs/song03_faith/tracks.png" style="zoom:70%" />
