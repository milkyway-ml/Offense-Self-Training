# Offense-Self-Training 
## Abstract
Online social media is rife with offensive and hateful comments, prompting the need of their automatic detection given the sheer amount of posts created at each second. However, collecting high-quality human-labeled datasets in order to train machine learning models for this task is difficult and costly, especially because (i) non-offensive posts are significantly more frequent than offensive ones and (ii) repeatedly exposing annotators to offensive content can be harmful to their mental health. Aiming to mitigate these limitations, this paper innovates by employing self-training, a semi-supervised technique that aggregates weakly-labeled instances to a human-labeled training set. We experiment with default self-training and the recent variation known as noisy student, which adds data augmentation during the self-training. Our results show that self-training significantly improves the model’s performance in three datasets offensive and hate speech detection, with different classification labels. Despite being successfully applied in other NLP tasks, noisy student does not consistently outperform the default self-training method in our experiments. We analyse the negative results obtained by the noisy student method, providing useful insights for other NLP tasks.



## How to reproduce the experiments:
* Make sure to use Python 3.10.4. 
* Using a decent GPU is heavily encouraged.
0. (Optional) Installing dependencies with conda:
    >conda create -n selftrain python==3.10.4

    >conda activate selftrain
1. Install python dependencies.
    >pip install -r requirements.txt
2. Move your current directory to the experiments folder.
    >cd experiments
3. Download the data sets.
    >make download-datasets
4. Run one of the experiments:
    >make experiment-name

    Where "experiment-name" should be one of the following:
    * olid-default
    * olid-ns
    * convabuse-default
    * convabuse-ns
    * mhs-default
    * mhs-ns


## Citing
tba.

## License
tba.