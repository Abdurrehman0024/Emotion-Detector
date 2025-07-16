function detectEmotion() {
  const text = document.getElementById('textInput').value;

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('result').innerText = `The above prompt shows Emotion: ${data.emotion}`;
    console.log("`Emotion: ${data.emotion}`;")
  })
  .catch(err => {
    console.error(err);
    document.getElementById('result').innerText = '❌ Error detecting emotion.';
  });
}

