# Spam Detection with Docker 🐋

### What’s this project about? 

This is my **Spam Detection** project. The goal of this simple project, is to detect spam messages using a **Naive Bayes classifier** and package it all up into a **Docker container**. 

During the development of my MSc Medulloblastoma Classification dissertation project, I couldn’t get Docker implemented, so I decided to explore Docker with this smaller project. This project helped me get a feel for and understand containerization.

### Why Spam Detection? 📨

Spam detection is great starting point for a NLP. Detecting it efficiently is a great way to understand text classification. By using a dataset of SMS messages, I trained a basic model to identify whether a message is spam or not. I also generated **randomly generated spam messages** like "Congratulations! You've won a prize!".

### How does it work? 🔍

1. **Data**: SMS data from [this dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv), which labels messages as either "ham" (not spam) or "spam".
2. **Model**: A **Naive Bayes** classifier. The model transforms the text data into a vectorized format using **TF-IDF (Term Frequency-Inverse Document Frequency)** and then classifies it. Using TF-IDF to convert text into numerical vectors that the model can understand.
3. **Docker**: I then wrap it all up in a Docker container to keep everything tidy and reproducible. 

### How to Run 🏃‍♂️

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-repo/spam-detector-docker.git
   cd spam-detector-docker

2. **Build the Docker image**:
   ```bash
   docker build -t spam-detection .

3. **Run the container**:
   ```bash
   docker run --rm spam-detection

4. **Make a prediction: You can make a quick prediction by running**:
   ```bash
   docker run --rm spam-detection python predict.py "Claim your free gift now!"


Why Docker? 🐋
I wanted to explore Docker because it's a powerful tool that makes projects easily reproducible and has become industry standard. You can package everything up into one container, stealing Java's mantra of write once run anywhere. During my dissertation, I struggled to implement Docker, so I decided to learn from scratch with this project. Now, I’ve taken my first step towards using Docker for machine learning models!

Future Plans 🚀
Next up, I want to explore CNNs and deep learning to expand on my original dissertation project, but for now, I’m happy to get more comfortable with Docker and traditional machine learning. One foot infront of the other! 💪
