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
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const uploadResult = document.getElementById('uploadResult');
    const formData = new FormData();

    if (fileInput.files.length === 0) {
        uploadResult.innerText = 'Please select a file first.';
        return;
    }

    formData.append('file', fileInput.files[0]);

    fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Received data: ', data); // For debugging
        if (data.sentiment) {
            uploadResult.innerText = `Sentiment: ${data.sentiment}`;
            up
        } else {
            uploadResult.innerText = 'Error uploading file.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        uploadResult.innerText = 'Error uploading file.';
    });
}
