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
  
  


function retrainModel() {
  document.getElementById('retrainButton').disabled = true;
  
  fetch('/api/retrain', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({label: 'New', proportionFactor: 2})
  })
  .then(response => {
      if (!response.ok) {
          throw new Error('Network response was not ok');
      }
      return response.json();
  })
  .then(data => {
      alert('Model retrained with new data!');
      if (data.modelName) {
        updatePlots(data.modelName);  // Use the updated model name from the response
    } else {
        throw new Error('Model name not provided in response');
    }
      // // Assume `data` includes the model identifier if necessary
      // const modelName = data.modelName || 'trained_sgd_model'; // Use a default if modelName is undefined
      // updatePlots(modelName);  // Pass the new model name to updatePlots
  })
  .catch(error => {
      console.error('Error during retraining:', error);
      alert('Failed to retrain model!');
  })
  .finally(() => {
      document.getElementById('retrainButton').disabled = false;
  });
}


function updatePlots(modelName) {
  fetch(`/api/plot-data?model=${modelName}`)  // Include the model identifier in the API request
  .then(response => {
      if (!response.ok) {
          throw new Error('Failed to fetch plot data');
      }
      return response.json();
  })
  .then(data => {
      const plotConfig = {
          x: data.actual,
          y: data.predicted,
          type: 'scatter',
          mode: 'markers',
          marker: {color: 'blue'}
      };
      Plotly.newPlot('plotDiv', [plotConfig], {
          title: 'Updated Actual vs. Predicted Prices',
          xaxis: {title: 'Actual Prices'},
          yaxis: {title: 'Predicted Prices'}
      });
  })
  .catch(error => {
      console.error('Error updating plots:', error);
      alert('Error fetching new plot data.');
  });
}
