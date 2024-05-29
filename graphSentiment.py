import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def plotSentimentBar(positiveNumber, negativeNumber):
    # Set up labels of positive and negative sentiments
    labels = ['Positive Reviews', 'Negative Reviews']
    counts = [positiveNumber, negativeNumber]
    # Bar chart creation
    # Narrower bars
    figure, axis = plt.subplots(figsize=(8,6))
    bars = axis.bar(labels, counts, color=['lime', 'red'])
    axis.set_facecolor('lightgrey')
    figure.patch.set_facecolor('skyblue')
    axis.set_xlabel('Sentiment',  fontdict={'fontsize': 16, 'fontweight': 'bold'})
    axis.set_ylabel('Number of Reviews',  fontdict={'fontsize': 16, 'fontweight': 'bold'})
    axis.set_title('Overall Product Sentiment', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2.0, 
            height, 
            f'{height}', 
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold'
        )
    plt.savefig('frontendUI/Bar_Graph.png')
    plt.close(figure)

def plotSentimentPie(positiveNumber, negativeNumber):
    # Set up pie chart
    sentLabels = ['Positive Reviews', 'Negative Reviews']
    counts = [positiveNumber, negativeNumber]
    # Set up plot subplots size
    figure, axis = plt.subplots(figsize=(8,6))
    axis.pie(counts, labels=sentLabels, colors = ['lime', 'red'], autopct='%1.1f%%', textprops={'fontsize': 14, 'fontweight': 'bold'})
    figure.patch.set_facecolor('skyblue')
    axis.set_title('Overall Product Sentiment', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.savefig('frontendUI/Pie_Chart.png')
    plt.close(figure)
    
