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
  let data = {
    label: 'New',
    proportionFactor: 2
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
                      <td>${model.name}</td>
                      <td>${model.version}</td>
                      <td>${model.run_id}</td>
                      <td>${model.status}</td>
                      <td>${JSON.stringify(model.metrics)}</td>
                  `;
                  tableBody.appendChild(row);
              }
          });
      })
      .catch(error => console.error('Failed to fetch model information:', error));
}

// Event listeners and initial fetch calls
document.addEventListener('DOMContentLoaded', function () {
  // Attach event listeners to form and button
  document.getElementById('diamond-form').addEventListener('submit', predictPrice);
  document.getElementById('retrainButton').addEventListener('click', retrainModel);
  // Fetch initial data to populate UI components
  fetchModels();
  fetchModelInfo();
});