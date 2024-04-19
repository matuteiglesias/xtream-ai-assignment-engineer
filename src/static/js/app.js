// src/static/js/app.js

  // Function to handle price prediction
function predictPrice(event) {
  event.preventDefault();
  let formData = new FormData(event.target);
  let data = {};  
  formData.forEach((value, key) => { data[key] = value; });
  let selectedModel = document.getElementById('model-select').value;
  data['model'] = selectedModel;

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data => alert('Predicted Price: ' + data.price))
  .catch(error => console.error('Error:', error));  
}



// Function to handle model retraining
function retrainModel() {

  let label = document.getElementById('label-input').value;
  let proportionFactor = parseFloat(document.getElementById('proportion-factor-input').value);
  let nSamples = parseInt(document.getElementById('sample-input').value);

  let data = {
    label: label,
    proportionFactor: proportionFactor,
    nSamples: nSamples
  };

  
  fetch('/api/retrain', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data => {   
    alert('Model retrained successfully: ' + data.modelName);
    fetchModels();  // Refresh the model list
    fetchModelInfo(); // Refresh the model details table
    updatePlots(data.run_id);  // Update plots with the new run_id   
  })
  .catch(error => {
    console.error('Error during retraining:', error);
    alert('Failed to retrain model!');
  })
  .finally(() => {
    document.getElementById('retrainButton').disabled = false;
  });
}


function updatePlots(runId) {
  console.log('Fetching plot data for run ID:', runId); // Log the run ID being used
  fetch(`/api/plot-data?run_id=${runId}`)
    .then(response => {   
      console.log('Received response from server'); // Log when the response is received
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);    
      }
      return response.json();
    })
    .then(data => {
      console.log('Processing data received from server:', data); // Log the data received
      let plotData = data.image; // Assuming the server returns a base64 encoded image
      let img = document.createElement('img');
      img.src = `data:image/png;base64,${plotData}`;
      let plotDiv = document.getElementById('plotDiv');
      plotDiv.innerHTML = ''; // Clear existing content
      plotDiv.appendChild(img);
      console.log('Plot image appended to the plot div.'); // Confirm the image is appended
    })
    .catch(error => {
      console.error('Error updating plots:', error); // Log any errors
      console.log('Failed to update plots for run ID:', runId);
    });
}



  // Function to fetch and display models in the dropdown
function fetchModels() {
  fetch('/api/models')
      .then(response => response.json())
      .then(data => {
          const select = document.getElementById('model-select');
          select.innerHTML = '';
          data.models.forEach(modelFilename => {
              const option = document.createElement('option');
              option.value = modelFilename;
              option.textContent = modelFilename;
              select.appendChild(option);
          });
      })
      .catch(error => console.error('Failed to load models:', error));
}

function fetchModelInfo() {
  fetch('/api/get-model-info')
      .then(response => response.json())
      .then(data => {
          const tableBody = document.getElementById('model-info');

          // Before appending, clear table if this is the first call
          if (!tableBody.hasAttribute('data-populated')) {
              tableBody.innerHTML = '';
              tableBody.setAttribute('data-populated', 'true');
          }

          data.forEach(model => {
              // Check if row with run_id already exists
              if (!document.querySelector(`#model-info [data-run-id="${model.run_id}"]`)) {
                  const row = document.createElement('tr');
                  row.setAttribute('data-run-id', model.run_id); // Set a data attribute for run_id
                  row.innerHTML = `
                      <td>${model.version}</td>
                      <td>${model.metrics.MAE.toFixed(3)}</td>
                      <td>${model.metrics.MSE.toFixed(3)}</td>
                      <td>${model.metrics.R2.toFixed(3)}</td>
                      <td>${model.metrics.RMSE.toFixed(3)}</td>                      
                  `;
                  tableBody.appendChild(row);
              }
          });
      })
      .catch(error => console.error('Failed to fetch model information:', error));
}


function performTest(endpoint) {
  axios.post('/api/test_performance', { endpoint: endpoint })
      .then(function (response) {
          document.getElementById('responseTime').innerText = 'Average Response Time: ' + response.data.response_time.toFixed(3) + ' seconds';
          document.getElementById('throughput').innerText = 'Throughput: ' + response.data.throughput.toFixed(3) + ' requests/second';
      })
      .catch(function (error) {
          console.error('Error during performance testing:', error);
      });
}



// Event listeners and initial fetch calls
document.addEventListener('DOMContentLoaded', function () {   ////////////
  // Attach event listeners to form and button
  document.getElementById('diamond-form').addEventListener('submit', predictPrice);
  document.getElementById('retrainButton').addEventListener('click', retrainModel);   ////////////
  // Fetch initial data to populate UI components
  fetchModels();
  fetchModelInfo();
  document.getElementById('testPredictBtn').addEventListener('click', function() {
    performTest('predict');
  });
  document.getElementById('testRetrainBtn').addEventListener('click', function() {
    performTest('retrain');
  });

  


});

