document.getElementById('diamond-form').onsubmit = function(event) {
    event.preventDefault();
    let formData = new FormData(event.target);  // Create a new FormData object
    let data = {};   // Create an empty object
    formData.forEach((value, key) => { data[key] = value; });   // Loop through the form data and populate the empty object
    let selectedModel = document.getElementById('model-select').value;
    data['model'] = selectedModel;  // Add the selected model to the data object


    // fetch('/predict', { method: 'POST', body: formData })
    fetch('/api/predict', {   // Use this URL for the Flask API
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })  
    .then(response => response.json()) // Parse the JSON response
    .then(data => alert('Predicted Price: ' + data.price)) // Display the prediction
    .catch(error => console.error('Error:', error));  
  };
  

  function fetchModels() {   // Fetch the list of models from the server
    fetch('/api/models')
      .then(response => response.json())
      .then(data => {
        const select = document.getElementById('model-select');
        select.innerHTML = '';
        data.models.forEach(modelFilename => {
            const option = document.createElement('option');
            option.value = modelFilename;  // Use the filename as the value
            option.textContent = modelFilename;  // Use the filename as the text content
            select.appendChild(option);
        });

        // data.models.forEach(model => {
        //   const option = document.createElement('option');
        //   option.value = model.id;
        //   option.textContent = model.name;
        //   select.appendChild(option);
        // });
      }).catch(error => console.error('Failed to load models:', error));
  }
  window.onload = fetchModels;
  
  

  // function retrainModel() {   // Retrain the selected model
  //   fetch('/api/retrain', { method: 'POST' })
  //     .then(response => response.json())
  //     .then(data => {
  //       alert('Model retrained!');
  //       fetchModels();  // Refresh model list and stats
  //     }).catch(error => console.error('Error during retraining:', error));
  // }
  

  function retrainModel() {
    // Send a POST request to trigger data addition and retraining
    fetch('/api/retrain', { 
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({label: 'New', proportionFactor: 2})
    })
    .then(response => response.json())
    .then(data => {
        alert('Model retrained with new data!');
        fetchModels(); // Refresh model list and stats
    })
    .catch(error => console.error('Error during retraining:', error));
}
