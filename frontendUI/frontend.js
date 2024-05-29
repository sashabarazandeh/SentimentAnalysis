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
function newPageToUpload(){
    window.location.href = 'uploadFile.html'
}
function backToIndex(){
    window.location.href = 'index.html'
}
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const uploadResult = document.getElementById('uploadResult');
    const formData = new FormData();
    const barPng = document.getElementById('barPng');
    const piePng = document.getElementById('piePng');


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
        if (data) {
            uploadResult.innerText = `Sentiments: ${data[1].positive}% positive, ${data[2].negative}% negative`;
            if (data[3].barPng) {
                barPng.src = `${data[3].barPng}`;
                barPng.style.display = 'block'
            }
            if (data[4].piePng){
                piePng.src = `${data[4].piePng}`;
                piePng.style.display='block';
            }
        } else {
            uploadResult.innerText = 'Error uploading file.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        uploadResult.innerText = 'Error uploading file.';
    });
}
