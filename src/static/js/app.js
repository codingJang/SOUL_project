// SOUL Project Web Interface JavaScript

class SOULApp {
    constructor() {
        this.websocket = null;
        this.charts = {};
        this.colors = null; // Will be loaded from backend
        this.colorScheme = null; // Will store full color scheme data
        this.items = [
            'interest_rates', 'gdp', 'dem_after_shock', 
            'price_lvl', 'delta_price_lvl', 'affinity', 'delta_affinity'
        ];
        this.currentState = {
            is_playing: false,
            step: 0,
            checkpoint_loaded: false
        };
        
        // Store time series data for real-time charts
        this.timeSeriesData = {
            'interest_rates': [],
            'gdp': [],
            'price_lvl': [],
            'affinity': []
        };
        
        // Speed multiplier for simulation
        this.speedMultiplier = 1;
        
        this.initializeApp();
    }

    async initializeApp() {
        this.initializeUI();
        await this.loadColorScheme();
        this.initializeCharts();
        this.loadCheckpoints();
        this.updateHistoryChart(); // Initialize historical chart
        this.connectWebSocket();
    }

    initializeUI() {
        // Get UI elements
        this.elements = {
            checkpointSelect: document.getElementById('checkpointSelect'),
            historySelect: document.getElementById('historySelect'),
            loadBtn: document.getElementById('loadBtn'),
            playBtn: document.getElementById('playBtn'),
            pauseBtn: document.getElementById('pauseBtn'),
            speedSelect: document.getElementById('speedSelect'),
            connectionStatus: document.getElementById('connectionStatus'),
            stepCounter: document.getElementById('stepCounter'),
            simulationStatus: document.getElementById('simulationStatus')
        };

        // Add event listeners
        this.elements.loadBtn.addEventListener('click', () => this.loadCheckpoint());
        this.elements.playBtn.addEventListener('click', () => this.startSimulation());
        this.elements.pauseBtn.addEventListener('click', () => this.pauseSimulation());
        this.elements.historySelect.addEventListener('change', () => this.updateHistoryChart());
        this.elements.speedSelect.addEventListener('change', () => this.updateSpeed());
        this.elements.checkpointSelect.addEventListener('change', () => {
            this.elements.loadBtn.disabled = !this.elements.checkpointSelect.value;
        });
    }

    async loadColorScheme() {
        try {
            const response = await fetch('/color_scheme');
            this.colorScheme = await response.json();
            this.colors = this.colorScheme.agent_colors;
            console.log('Color scheme loaded:', this.colorScheme);
        } catch (error) {
            console.error('Error loading color scheme, using fallback:', error);
            // Fallback colors if backend fails
            this.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'];
        }
    }

    initializeCharts() {
        if (!this.colors) {
            console.error('Colors not loaded yet!');
            return;
        }

        // Chart configuration for real-time charts
        const realtimeChartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0 // Disable animations for real-time updates
                },
                elements: {
                    line: {
                        tension: 0, // Ensure linear interpolation
                        stepped: false
                    },
                    point: {
                        radius: 1
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Time Step'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        };

        // Initialize real-time time series charts (excluding affinity matrix)
        const timeSeriesChartIds = ['interestRatesChart', 'gdpChart', 'priceLvlChart'];
        timeSeriesChartIds.forEach(chartId => {
            const ctx = document.getElementById(chartId).getContext('2d');
            
            // Create datasets for each agent using consistent colors
            const datasets = [];
            for (let i = 0; i < this.colors.length; i++) {
                datasets.push({
                    label: `Agent ${i}`,
                    data: [],
                    borderColor: this.colors[i],
                    backgroundColor: this.colors[i] + '20',
                    tension: 0,
                    stepped: false, // Ensure linear interpolation
                    pointRadius: 1
                });
            }
            
            this.charts[chartId] = new Chart(ctx, {
                ...realtimeChartConfig,
                data: { datasets }
            });
        });

        // Initialize affinity matrix table
        this.initializeAffinityTable();

        // Initialize history chart
        const historyCtx = document.getElementById('historyChart').getContext('2d');
        this.charts.historyChart = new Chart(historyCtx, {
            type: 'line',
            data: {
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Time Step'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    initializeAffinityTable() {
        // Table is already initialized in HTML, just store reference
        this.affinityTable = document.getElementById('affinityTable');
    }

    async loadCheckpoints() {
        try {
            const response = await fetch('/checkpoints');
            const checkpoints = await response.json();
            
            const select = this.elements.checkpointSelect;
            select.innerHTML = '<option value="">Select a checkpoint...</option>';
            
            checkpoints.forEach(checkpoint => {
                const option = document.createElement('option');
                option.value = checkpoint.full_path;
                option.textContent = checkpoint.display_name;
                select.appendChild(option);
            });
            
            if (checkpoints.length > 0) {
                // Select the most recent checkpoint by default
                select.value = checkpoints[0].full_path;
                this.elements.loadBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error loading checkpoints:', error);
            this.showNotification('Error loading checkpoints', 'error');
        }
    }

    async loadCheckpoint() {
        const checkpointPath = this.elements.checkpointSelect.value;
        if (!checkpointPath) return;

        this.setLoading(true);
        
        try {
            const response = await fetch('/load_checkpoint', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `checkpoint_path=${encodeURIComponent(checkpointPath)}`
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('Checkpoint loaded successfully', 'success');
                this.elements.playBtn.disabled = false;
                this.currentState.checkpoint_loaded = true;
                this.clearTimeSeriesData(); // Reset charts for new checkpoint
                this.updateHistoryChart(); // Update historical chart with new data
                this.updateStatus();
            } else {
                this.showNotification('Error loading checkpoint', 'error');
            }
        } catch (error) {
            console.error('Error loading checkpoint:', error);
            this.showNotification('Error loading checkpoint', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    async startSimulation() {
        try {
            const response = await fetch('/start_simulation', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.currentState.is_playing = true;
                this.updateStatus();
                this.elements.playBtn.disabled = true;
                this.elements.pauseBtn.disabled = false;
                this.showNotification('Simulation started', 'success');
            }
        } catch (error) {
            console.error('Error starting simulation:', error);
            this.showNotification('Error starting simulation', 'error');
        }
    }

    async pauseSimulation() {
        try {
            const response = await fetch('/pause_simulation', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.currentState.is_playing = false;
                this.updateStatus();
                this.elements.playBtn.disabled = false;
                this.elements.pauseBtn.disabled = true;
                this.showNotification('Simulation paused', 'info');
            }
        } catch (error) {
            console.error('Error pausing simulation:', error);
            this.showNotification('Error pausing simulation', 'error');
        }
    }

    async updateSpeed() {
        try {
            this.speedMultiplier = parseFloat(this.elements.speedSelect.value);
            const response = await fetch('/set_speed', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ speed: this.speedMultiplier })
            });
            const result = await response.json();
            
            if (result.success) {
                this.showNotification(`Speed set to ${this.speedMultiplier}x`, 'info');
            }
        } catch (error) {
            console.error('Error setting speed:', error);
            this.showNotification('Error setting speed', 'error');
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
    }

    handleWebSocketMessage(data) {
        if (data.type === 'simulation_data') {
            this.updateCharts(data.data);
        } else {
            // State update
            this.currentState = { ...this.currentState, ...data };
            this.updateStatus();
            
            if (data.current_data) {
                this.updateCharts(data.current_data);
            }
        }
    }

    updateCharts(data) {
        // Update time series charts
        const timeSeriesChartMappings = {
            'interestRatesChart': 'interest_rates',
            'gdpChart': 'gdp',
            'priceLvlChart': 'price_lvl'
        };

        const currentStep = this.currentState.step;

        // Update time series charts
        Object.entries(timeSeriesChartMappings).forEach(([chartId, dataKey]) => {
            if (data[dataKey] && this.charts[chartId]) {
                const chart = this.charts[chartId];
                const values = data[dataKey];
                
                // Store time series data for this metric
                if (!this.timeSeriesData[dataKey]) {
                    this.timeSeriesData[dataKey] = [];
                }
                
                // Add new data point for this time step
                this.timeSeriesData[dataKey].push({
                    step: currentStep,
                    values: values
                });
                
                // Update chart datasets - one for each agent (no sliding window)
                values.forEach((value, agentIndex) => {
                    if (chart.data.datasets[agentIndex]) {
                        // Add new data point
                        chart.data.datasets[agentIndex].data.push({
                            x: currentStep,
                            y: value
                        });
                    }
                });
                
                chart.update('none'); // Update without animation
            }
        });

        // Update affinity matrix table separately
        if (data.affinity && this.affinityTable) {
            this.updateAffinityTable(data.affinity);
        }
        
        // Update historical chart in real-time
        this.updateHistoryChart();
    }

    updateAffinityTable(affinityMatrix) {
        if (!this.affinityTable) return;
        
        // Find min and max values for color scaling
        let minValue = Math.min(...affinityMatrix.flat());
        let maxValue = Math.max(...affinityMatrix.flat());
        
        // Update each cell in the table
        for (let i = 0; i < affinityMatrix.length; i++) {
            for (let j = 0; j < affinityMatrix[i].length; j++) {
                const cell = this.affinityTable.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                if (cell) {
                    const value = affinityMatrix[i][j];
                    
                    // Update cell content
                    cell.textContent = value.toFixed(3);
                    
                    // Add tooltip
                    cell.title = `Agent ${i} → Agent ${j}: ${value.toFixed(3)}`;
                    
                    // Color coding based on value
                    // Map value from [minValue, maxValue] to [0, 10] for CSS class
                    const normalizedValue = maxValue > minValue ? (value - minValue) / (maxValue - minValue) : 0;
                    const colorIndex = Math.round(normalizedValue * 10);
                    
                    // Remove existing affinity classes
                    cell.className = cell.className.replace(/affinity-\d+/g, '');
                    
                    // Add new classes
                    if (i === j) {
                        cell.classList.add('matrix-cell', 'diagonal', `affinity-${colorIndex}`);
                    } else {
                        cell.classList.add('matrix-cell', `affinity-${colorIndex}`);
                    }
                }
            }
        }
    }

    async updateHistoryChart() {
        const selectedItem = this.elements.historySelect.value;
        
        if (!this.colors) {
            console.error('Colors not loaded yet for history chart');
            return;
        }
        
        try {
            const response = await fetch(`/history/${selectedItem}`);
            const historyData = await response.json();
            
            if (historyData[selectedItem]) {
                const chart = this.charts.historyChart;
                const agentData = historyData[selectedItem];
                
                // Clear existing datasets
                chart.data.datasets = [];
                
                // Create a dataset for each agent using consistent colors
                agentData.forEach((agentHistory, agentIndex) => {
                    if (agentHistory.length > 0) {
                        const color = this.colors[agentIndex % this.colors.length];
                        chart.data.datasets.push({
                            label: `Agent ${agentIndex}`,
                            data: agentHistory.map((value, timeStep) => ({
                                x: timeStep,
                                y: value
                            })),
                            borderColor: color,
                            backgroundColor: color + '20',
                            tension: 0,
                            stepped: false, // Ensure linear interpolation
                            pointRadius: 1
                        });
                    }
                });
                
                chart.update('none');
            }
        } catch (error) {
            console.error('Error updating history chart:', error);
        }
    }

    updateConnectionStatus(connected) {
        const status = this.elements.connectionStatus;
        if (connected) {
            status.innerHTML = '<span class="status-indicator"></span>Connected';
            status.className = 'badge bg-success status-connected';
        } else {
            status.innerHTML = '<span class="status-indicator"></span>Disconnected';
            status.className = 'badge bg-danger status-disconnected';
        }
    }

    updateStatus() {
        // Update step counter
        this.elements.stepCounter.textContent = `Step: ${this.currentState.step}`;
        
        // Update simulation status
        const status = this.elements.simulationStatus;
        if (this.currentState.is_playing) {
            status.innerHTML = '<span class="status-indicator"></span>Running';
            status.className = 'badge bg-success status-playing';
        } else if (this.currentState.checkpoint_loaded) {
            status.innerHTML = '<span class="status-indicator"></span>Paused';
            status.className = 'badge bg-warning status-paused';
        } else {
            status.innerHTML = '<span class="status-indicator"></span>Stopped';
            status.className = 'badge bg-secondary status-stopped';
        }
        
        // Update button states
        if (this.currentState.is_terminated) {
            this.elements.playBtn.disabled = true;
            this.elements.pauseBtn.disabled = true;
            this.showNotification('Simulation terminated', 'warning');
        }
    }

    setLoading(loading) {
        const buttons = [this.elements.loadBtn, this.elements.playBtn, this.elements.pauseBtn];
        buttons.forEach(btn => {
            if (loading) {
                btn.classList.add('loading');
                btn.disabled = true;
            } else {
                btn.classList.remove('loading');
                // Re-enable based on current state
                this.elements.loadBtn.disabled = !this.elements.checkpointSelect.value;
                this.elements.playBtn.disabled = !this.currentState.checkpoint_loaded || this.currentState.is_playing;
                this.elements.pauseBtn.disabled = !this.currentState.is_playing;
            }
        });
    }

    clearTimeSeriesData() {
        // Clear stored time series data
        Object.keys(this.timeSeriesData).forEach(key => {
            this.timeSeriesData[key] = [];
        });
        
        // Clear time series chart data
        const timeSeriesChartIds = ['interestRatesChart', 'gdpChart', 'priceLvlChart'];
        timeSeriesChartIds.forEach(chartId => {
            if (this.charts[chartId]) {
                this.charts[chartId].data.datasets.forEach(dataset => {
                    dataset.data = [];
                });
                this.charts[chartId].update('none');
            }
        });

        // Clear affinity matrix table
        if (this.affinityTable) {
            const cells = this.affinityTable.querySelectorAll('.matrix-cell');
            cells.forEach(cell => {
                cell.textContent = '-';
                cell.title = '';
                cell.className = 'matrix-cell';
            });
        }
    }

    showNotification(message, type = 'info') {
        // Simple notification system
        const notification = document.createElement('div');
        notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    window.soulApp = new SOULApp();
}); 