import React, { useState, useEffect } from 'react';
import { Box, Typography, CircularProgress, Paper, Grid, Tooltip } from '@mui/material';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip as ChartTooltip, Legend } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend
);

const InteractiveVisualization = ({ 
  type = 'line', // 'line', 'bar', 'neural-network'
  title = 'Data Visualization',
  description = 'Interactive visualization of data',
  data = null,
  options = {},
  loading = false,
  error = null,
  height = 400,
  width = '100%'
}) => {
  const [chartData, setChartData] = useState(null);
  
  useEffect(() => {
    if (data) {
      setChartData(data);
    } else {
      // Generate sample data if none provided
      generateSampleData();
    }
  }, [data, type]);
  
  const generateSampleData = () => {
    const labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July'];
    
    if (type === 'line') {
      setChartData({
        labels,
        datasets: [
          {
            label: 'Dataset 1',
            data: labels.map(() => Math.random() * 1000),
            borderColor: 'rgb(53, 162, 235)',
            backgroundColor: 'rgba(53, 162, 235, 0.5)',
          },
          {
            label: 'Dataset 2',
            data: labels.map(() => Math.random() * 1000),
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
          },
        ],
      });
    } else if (type === 'bar') {
      setChartData({
        labels,
        datasets: [
          {
            label: 'Dataset 1',
            data: labels.map(() => Math.random() * 1000),
            backgroundColor: 'rgba(53, 162, 235, 0.5)',
          },
          {
            label: 'Dataset 2',
            data: labels.map(() => Math.random() * 1000),
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
          },
        ],
      });
    } else if (type === 'neural-network') {
      // For neural network visualization, we'll use a custom renderer
      setChartData({
        layers: [
          { nodes: 4, name: 'Input Layer' },
          { nodes: 8, name: 'Hidden Layer 1' },
          { nodes: 6, name: 'Hidden Layer 2' },
          { nodes: 2, name: 'Output Layer' }
        ],
        connections: true
      });
    }
  };
  
  const defaultOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: title,
      },
    },
  };
  
  const mergedOptions = { ...defaultOptions, ...options };
  
  const renderNeuralNetwork = () => {
    if (!chartData || !chartData.layers) return null;
    
    const layers = chartData.layers;
    const maxNodes = Math.max(...layers.map(layer => layer.nodes));
    const nodeSize = 30;
    const nodeSpacing = 20;
    const layerSpacing = 150;
    
    return (
      <Box sx={{ 
        width: '100%', 
        height: height, 
        position: 'relative',
        overflow: 'auto'
      }}>
        <svg 
          width={layers.length * layerSpacing} 
          height={maxNodes * (nodeSize + nodeSpacing)}
          style={{ margin: '0 auto', display: 'block' }}
        >
          {/* Draw connections between nodes */}
          {chartData.connections && layers.slice(0, -1).map((layer, layerIndex) => {
            const nextLayer = layers[layerIndex + 1];
            const connections = [];
            
            for (let i = 0; i < layer.nodes; i++) {
              const x1 = layerIndex * layerSpacing + nodeSize;
              const y1 = i * (nodeSize + nodeSpacing) + nodeSize / 2;
              
              for (let j = 0; j < nextLayer.nodes; j++) {
                const x2 = (layerIndex + 1) * layerSpacing;
                const y2 = j * (nodeSize + nodeSpacing) + nodeSize / 2;
                
                connections.push(
                  <line 
                    key={`${layerIndex}-${i}-${j}`}
                    x1={x1} 
                    y1={y1} 
                    x2={x2} 
                    y2={y2} 
                    stroke="rgba(0, 0, 0, 0.2)" 
                    strokeWidth="1"
                  />
                );
              }
            }
            
            return connections;
          })}
          
          {/* Draw nodes */}
          {layers.map((layer, layerIndex) => {
            const nodes = [];
            
            for (let i = 0; i < layer.nodes; i++) {
              const cx = layerIndex * layerSpacing + nodeSize / 2;
              const cy = i * (nodeSize + nodeSpacing) + nodeSize / 2;
              
              nodes.push(
                <g key={`node-${layerIndex}-${i}`}>
                  <circle 
                    cx={cx} 
                    cy={cy} 
                    r={nodeSize / 2} 
                    fill={
                      layerIndex === 0 ? 'rgba(53, 162, 235, 0.7)' : 
                      layerIndex === layers.length - 1 ? 'rgba(255, 99, 132, 0.7)' : 
                      'rgba(75, 192, 192, 0.7)'
                    } 
                    stroke="#fff"
                    strokeWidth="2"
                  />
                  <text 
                    x={cx} 
                    y={cy} 
                    textAnchor="middle" 
                    dominantBaseline="middle"
                    fill="#fff"
                    fontSize="12"
                  >
                    {layerIndex === 0 ? `I${i+1}` : 
                     layerIndex === layers.length - 1 ? `O${i+1}` : 
                     `H${layerIndex}${i+1}`}
                  </text>
                </g>
              );
            }
            
            // Layer labels
            nodes.push(
              <text 
                key={`layer-${layerIndex}`}
                x={layerIndex * layerSpacing + nodeSize / 2} 
                y={layer.nodes * (nodeSize + nodeSpacing) + 20} 
                textAnchor="middle"
                fill="#333"
                fontWeight="bold"
              >
                {layer.name}
              </text>
            );
            
            return nodes;
          })}
        </svg>
      </Box>
    );
  };
  
  if (loading) {
    return (
      <Box 
        sx={{ 
          height: height, 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center' 
        }}
      >
        <CircularProgress />
      </Box>
    );
  }
  
  if (error) {
    return (
      <Box 
        sx={{ 
          height: height, 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center' 
        }}
      >
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }
  
  if (!chartData) {
    return (
      <Box 
        sx={{ 
          height: height, 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center' 
        }}
      >
        <Typography>No data available</Typography>
      </Box>
    );
  }
  
  return (
    <Paper sx={{ p: 3, mb: 4, width: width }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      
      {description && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {description}
        </Typography>
      )}
      
      <Box sx={{ height: height }}>
        {type === 'line' && <Line options={mergedOptions} data={chartData} />}
        {type === 'bar' && <Bar options={mergedOptions} data={chartData} />}
        {type === 'neural-network' && renderNeuralNetwork()}
      </Box>
    </Paper>
  );
};

// Example usage component with multiple visualizations
const VisualizationDashboard = () => {
  return (
    <Box sx={{ my: 4 }}>
      <Typography variant="h5" gutterBottom>
        Interactive AI Visualizations
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <InteractiveVisualization 
            type="neural-network"
            title="Neural Network Architecture"
            description="Interactive visualization of a neural network with multiple layers"
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <InteractiveVisualization 
            type="line"
            title="Training and Validation Loss"
            description="Model performance over training epochs"
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <InteractiveVisualization 
            type="bar"
            title="Model Accuracy by Class"
            description="Classification accuracy for different categories"
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export { InteractiveVisualization, VisualizationDashboard };
