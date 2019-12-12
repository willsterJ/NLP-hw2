# NLP-hw2

script.py is the main class. Run the following command:

```
pip3 install -r requirements.txt
python3 ./main.py --model [see script.py file for models] --lr [learning rate] --lamb[lambda] --epsilon[epsilon threshold] --t[epoch max]
```

By the default, without entering arguments, the program executes:
- model : unigram
- lr : 0.5
- lamb : 0.1
- epsilon : 0.00005
- t : 500
