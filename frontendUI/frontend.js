function analyzeSentiment() {
    const textInput = document.getElementById('textInput').value;
    const resultDiv = document.getElementById('result');

    fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: textInput }),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Recieved data: ', data); //remove later
        console.log('Sentiment of data: ', data.sentiment); //remove later
        if (data.sentiment){
            resultDiv.innerText = `Sentiment: ${data.sentiment}`;
        }else{
            resultDiv.innerText = `Error analyzing sentiment, ${data.sentiment} null`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultDiv.innerText = 'Error analyzing sentiment';
    });

}
