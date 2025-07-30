# SOUL Project Web Interface

The SOUL Project now includes a modern web interface that allows you to run economic simulations through your browser. This web interface is built with FastAPI and provides real-time visualization of simulation data.

## Features

- **Modern Web UI**: Clean, responsive interface built with Bootstrap
- **Real-time Charts**: Live visualization using Chart.js
- **WebSocket Connection**: Real-time updates without page refresh
- **Checkpoint Management**: Easy loading and selection of AI model checkpoints
- **Historical Data**: View historical trends for all simulation metrics
- **REST API**: Full RESTful API for programmatic access

## Getting Started

### Prerequisites

Ensure you have the project dependencies installed:

```bash
uv sync
```

### Starting the Web Application

#### Using the Startup Script (Recommended)

```bash
./scripts/run_webapp.sh
```

#### Manual Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the web application
uvicorn src.webapp:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the Interface

Once started, the web interface will be available at:

- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Using the Web Interface

### 1. Loading Checkpoints

1. The interface will automatically load available checkpoints from the `./models/` directory
2. Select a checkpoint from the dropdown menu
3. Click "Load Checkpoint" to initialize the simulation environment
4. Wait for the confirmation message

### 2. Running Simulations

1. Once a checkpoint is loaded, click "Start" to begin the simulation
2. The simulation will run continuously, updating charts in real-time
3. Use "Pause" to stop the simulation temporarily
4. The step counter shows the current simulation step

### 3. Viewing Data

#### Real-time Charts
- **Interest Rates**: Current interest rate values for all agents
- **GDP**: Gross domestic product values
- **Price Level**: Economic price levels
- **Affinity**: Agent affinity values

#### Historical Analysis
- Use the "History View" dropdown to select different metrics
- The historical chart shows trends over time for all agents
- Each agent is represented by a different colored line

### 4. Connection Status

The interface displays several status indicators:
- **Connection Status**: WebSocket connection to the server
- **Step Counter**: Current simulation step
- **Simulation Status**: Running, Paused, or Stopped

## API Endpoints

The web application provides a full REST API:

### Checkpoints
- `GET /checkpoints` - List available checkpoints
- `POST /load_checkpoint` - Load a specific checkpoint

### Simulation Control
- `POST /start_simulation` - Start the simulation
- `POST /pause_simulation` - Pause the simulation
- `GET /simulation_state` - Get current simulation state

### Data Access
- `GET /history` - Get all historical data
- `GET /history/{item}` - Get historical data for specific metric

### WebSocket
- `WS /ws` - Real-time simulation data stream

## Technical Architecture

### Backend (FastAPI)
- **SimulationManager**: Handles AI model loading and simulation execution
- **ConnectionManager**: Manages WebSocket connections for real-time updates
- **RESTful API**: Provides programmatic access to all functionality

### Frontend (JavaScript/HTML)
- **Bootstrap UI**: Modern, responsive interface
- **Chart.js**: Real-time data visualization
- **WebSocket Client**: Live updates without page refresh
- **Fetch API**: REST API communication

### Real-time Communication
- WebSocket connection for live simulation data
- Automatic reconnection on connection loss
- JSON message format for data exchange

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'configs'
```
**Solution**: Ensure you're running from the project root directory with the virtual environment activated.

**2. Connection Refused**
```
curl: (7) Failed to connect to localhost port 8000
```
**Solution**: Check that the uvicorn server is running and the virtual environment is activated.

**3. No Checkpoints Found**
```
No checkpoints found in ./models/ directory!
```
**Solution**: Ensure you have trained models in the `./models/` directory or train new models first.

### Performance Tips

1. **Disable Browser Extensions**: Some ad blockers may interfere with WebSocket connections
2. **Use Chrome/Firefox**: For best Chart.js performance
3. **Close Unused Tabs**: To ensure adequate system resources
4. **Check Network**: Ensure localhost connections are not blocked by firewall

## Development

### Adding New Metrics

To add new simulation metrics to the web interface:

1. **Backend**: Update the `items` list in `SimulationManager`
2. **Frontend**: Add new chart containers to `index.html`
3. **JavaScript**: Update chart mappings in `app.js`
4. **Styling**: Add styles for new charts in `style.css`

### Customizing Charts

Chart.js configuration can be modified in `src/static/js/app.js`:

```javascript
const chartConfig = {
    type: 'line',
    options: {
        // Your custom options here
    }
};
```

### API Extensions

To add new API endpoints, extend the FastAPI application in `src/webapp.py`:

```python
@app.get("/my_new_endpoint")
async def my_new_endpoint():
    return {"message": "Hello World"}
```

## Security Considerations

- The web interface binds to `0.0.0.0:8000` by default for accessibility
- For production deployment, consider:
  - Using HTTPS with proper SSL certificates
  - Implementing authentication if needed
  - Configuring proper CORS policies
  - Running behind a reverse proxy (nginx, Apache)

## Related Documentation

- [Configuration Guide](configuration.md) - Environment and model configuration
- [Data Analysis](data-analysis.md) - Understanding simulation outputs
- [Color Schemes](color-schemes.md) - Visualization color configuration 