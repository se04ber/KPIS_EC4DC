# Discussed Ideas

Training/Retraining Encoder for different architectures and compare:
1. Train the full Encoder on the dataset along with the classifier
2. Train some part of the Encoder or train additoinal weights[its called LoRA which is Low Rank Adaptation]
3. The final approach would be to freeze the encoder and only train the classifier.

Just Inference performance comparison:
4. Of different pretrained encoders
5. Of encoder vs. full small or commercial LMs

Further Ideas:
6. Trying to see if we can analyse/understand something in the vector space of Encoder for (document classification)
(We also discussed about understanding how information is actually stored in the vector space. But it would probably be too much for this course :( )

More ideas can be added, if not mentioned......

# Suggestions / Things to Keep In Mind
- Professor also mentioned/emphasized on the preprocessing and the quality of the data on which we are training. TBH, we didn't thought about it during our discussion, but its quite important.
Yes therefore maybe if we do just a inference analysis we need to optimal find encoders trained mostly on the same dataset, if possible.

- Encoder models can take an input of specific size for example 512. (document chunks sizes, rechunking maybe needed)

- We would have to see how much compute power we have and how much is required for our approach. Current recommended environment by claude is Kaggle


# Ressources available on Kaggle:
## GPU compute:
* fixed weekly allowance of 
* guaranteed 30 GPU hours/per week (T4 or P100)
* 20h/week on TPU (v6e-8 (8 cores)?)
* we can use both in the same week
(source: currently only Claude, needs to be checked on Kaggle again)

## Storage:
* 20 GB, persistent
* per session ca. 29–30 GB of system RAM and 4 CPU cores
(source: currently only Claude, needs to be checked on Kaggle again)
